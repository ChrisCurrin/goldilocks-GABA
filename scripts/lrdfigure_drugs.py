import time
from collections import OrderedDict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import second
from matplotlib import patheffects
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle

import settings
from core.analysis import burst_stats, get_duration_interval, inst_burst_rate
from core.lrdfigure import MultiRunFigure
from settings import constants, logging, time_unit
from style.axes import adjust_spines, use_scalebar
from style.color import get_benzo_color, get_drug_label
from style.figure import new_gridspec
from style.plot_trace import plot_state_colorbar

logger = logging.getLogger(__name__)


class Drugs(MultiRunFigure):
    fig_name = "figure_2_drugs"
    monitors = {
        "r_all": True,
        "sp_all": False,
        "state_mon": ["E_GABA", "g_GABA", "I_GABA_rec"],
        "synapse_mon": False,
    }

    def __init__(
        self,
        benzo_strengths=(0, 0.5, 1, 2, 5, 10),
        E_Cl_0s=(-88, -60),
        duration=600,
        **kwargs,
    ):
        self.benzo_strengths: Iterable[float] = benzo_strengths
        self.E_Cl_0s = E_Cl_0s
        self.duration = duration
        super().__init__(
            OrderedDict(
                benzo_strength={"range": benzo_strengths, "title": "drug"},
                E_Cl_0={"range": E_Cl_0s, "title": "E_Cl_0"},
                E_Cl_end="E_Cl_0",
            ),
            default_params=dict(
                manual_cl=True,
                benzo_onset_t=duration / 2,
                benzo_off_t=duration,
                duration=duration,
            ),
            **kwargs,
        )

    def plot(
        self,
        timeit=True,
        split=True,
        vars_to_plot=("E_GABA", "r"),
        var_heights=None,
        stats_to_plot=("Number", "Amplitude", "Duration"),
        stats_heights=None,
        markersize=4,
        **kwargs,
    ):
        super().plot(**kwargs)
        logger.info("plotting")
        plot_time_start = time.time()
        drugs = self.benzo_strengths
        np_drugs = np.array(drugs)

        var_height_ratios = []
        if var_heights is None:
            var_heights = {"E_GABA": 1.5, "I_GABA": 2, "r": 5}
        for var in vars_to_plot:
            var_height_ratios.append(var_heights[var])
        T = np.round(self.df.index.values[-1])
        benzo_onset_t = T / 2
        benzo_off_t = T
        # bins = [0, benzo_onset_t, T]
        picro_drugs = np_drugs[np_drugs < 1]
        benzo_drugs = np_drugs[np_drugs > 1]
        bs_to_plot = []
        if len(picro_drugs):
            bs_to_plot.append(picro_drugs[0])
        if len(benzo_drugs):
            bs_to_plot.append(benzo_drugs[-1])

        plot_egaba = "E_GABA" in vars_to_plot

        all_stats = ("Number", "Amplitude", "Duration", "Interval")
        if stats_heights is None:
            stats_heights = {"Number": 1, "Amplitude": 1, "Duration": 1, "Interval": 1}
            stats_heights = [stats_heights[k] for k in stats_to_plot]
        stats_data = pd.DataFrame(
            columns=pd.MultiIndex.from_product([self.E_Cl_0s, all_stats]), index=drugs
        )
        gridspec = {
            "height_ratios": [5]
            + [3]  # explanatory examples
            + [10],  # more examples  # population stats
            "width_ratios": [6] * len(self.E_Cl_0s),  # column per E start
        }
        fig, gs = new_gridspec(
            nrows=len(gridspec["height_ratios"]),
            ncols=len(gridspec["width_ratios"]),
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL),
            grid_kwargs=gridspec,
        )
        gs.update(top=0.98, right=0.98, hspace=0.3, wspace=0.2)

        # create explanatory gridspecs
        sub_gs_var = {bs: {} for bs in bs_to_plot}
        sub_gs_var[1] = {}
        sub_gs_stats = {}
        ax_egabas = {}
        if split:
            gs_explan = GridSpecFromSubplotSpec(
                plot_egaba + 1,
                len(self.E_Cl_0s),
                subplot_spec=gs[0, :],
                hspace=0.0,
                height_ratios=list([0.15] if plot_egaba else []) + [1],
            )

            for j, ecl in enumerate(self.E_Cl_0s):
                # one egaba trace
                if plot_egaba:
                    ax_egabas[ecl] = fig.add_subplot(gs_explan[0, j])
                n_rows = len(bs_to_plot) * 4
                gs_blocks = GridSpecFromSubplotSpec(
                    n_rows,
                    2,
                    subplot_spec=gs_explan[plot_egaba, j],
                    hspace=0.1,
                    wspace=0.15,
                )
                mid = height = n_rows // 2
                s, e = mid - height // 2, mid + height // 2
                sub_gs_var[1][ecl] = GridSpecFromSubplotSpec(
                    len(vars_to_plot),
                    1,
                    subplot_spec=gs_blocks[s:e, 0],
                    hspace=0.0,
                    height_ratios=var_height_ratios,
                )
                i_bs = range(0, len(bs_to_plot) * 3, height)
                for i, bs in zip(i_bs, bs_to_plot):
                    gs_traces = GridSpecFromSubplotSpec(
                        len(vars_to_plot),
                        1,
                        subplot_spec=gs_blocks[i : i + height, -1],  # noqa: E203
                        hspace=0.0,
                        height_ratios=var_height_ratios,
                    )
                    sub_gs_var[bs][ecl] = gs_traces
        else:
            gs_explan = GridSpecFromSubplotSpec(
                len(bs_to_plot), len(self.E_Cl_0s), subplot_spec=gs[0, :], hspace=0.15
            )

            for i, bs in enumerate(bs_to_plot):
                if bs not in sub_gs_var:
                    sub_gs_var[bs] = {}
                    sub_gs_stats[bs] = {}
                for j, ecl in enumerate(self.E_Cl_0s):
                    sub_gs_var[bs][ecl] = GridSpecFromSubplotSpec(
                        len(vars_to_plot),
                        1,
                        subplot_spec=gs_explan[i, j],
                        hspace=0.05,
                        height_ratios=var_height_ratios,
                    )

        self.fig = fig

        rates_all: pd.DataFrame = self.df.xs("r_all", axis=1, level="var")
        _ax_vars = {}
        _ax_stats = {}
        for e, ecl in enumerate(self.E_Cl_0s):
            burst_colors = []
            # trace examples for more conditions
            gs_traces = GridSpecFromSubplotSpec(
                1, len(drugs), subplot_spec=gs[1, e], wspace=0.15
            )
            _ax_trace = None

            if split:
                self.plot_vars(
                    fig,
                    _ax_vars,
                    sub_gs_var,
                    benzo_off_t,
                    benzo_onset_t,
                    1,
                    ecl,
                    vars_to_plot,
                    ax_egabas,
                    split,
                )

            max_rates_e = np.max(rates_all.xs(ecl, axis=1, level="E_Cl_0").values)
            num_d = {}
            for d, drug in enumerate(drugs):
                logger.info(f"drug = {drug} \t ecl = {ecl}")
                rates: pd.Series = rates_all[drug, ecl]
                if isinstance(rates, pd.DataFrame):
                    # rates is DataFrame if there're duplicate keys *at all* in the drugs column
                    idx = num_d[drug] = num_d.setdefault(drug, -1) + 1
                    rates = rates.iloc[:, idx]
                rates = rates[
                    rates.index >= benzo_onset_t
                ]  # restrict to onset only (if onset is not at t=0)
                color, ampa_color = get_benzo_color(drug)
                if ampa_color is not None:
                    color = ampa_color
                burst_colors.append(color)
                if drug in bs_to_plot:
                    # get and plot variables
                    self.plot_vars(
                        fig,
                        _ax_vars,
                        sub_gs_var,
                        benzo_off_t,
                        benzo_onset_t,
                        drug,
                        ecl,
                        vars_to_plot,
                        ax_egabas,
                        split,
                    )
                _ax_trace = fig.add_subplot(gs_traces[d], sharex=_ax_trace)
                _ax_trace.set_ylim(0, max_rates_e)
                # plot population rate and calculate stats
                self.calc_stats_with_trace(
                    _ax_trace, benzo_off_t, benzo_onset_t, drug, ecl, rates, stats_data
                )

            # restrict view to only when drug is active
            _ax_trace.set_xlim(benzo_onset_t, T)
            # _ax_trace.set_ylim(0, _ax_trace.get_ylim()[1]*1.1)

            # plot stats
            x_axis = drugs  # xaxis is drug value
            x_axis = list(
                range(len(drugs))
            )  # x axis is evenly distributed by condition
            gs_stats = GridSpecFromSubplotSpec(
                len(stats_to_plot),
                1,
                subplot_spec=gs[2, e],
                hspace=0.2,
                height_ratios=stats_heights,
            )
            ax_stats = self.plot_stats(
                drugs,
                e,
                ecl,
                fig,
                gs_stats,
                markersize,
                stats_data,
                stats_to_plot,
                x_axis,
                _ax_stats,
            )
            ax_stats[0].set_xlim(-0.5, len(drugs) - 0.5)
            if e != 0:
                for _ax_stat in ax_stats:
                    _ax_stat.set_ylabel("")
            else:
                fig.align_ylabels(ax_stats)

        for key, _ax in _ax_vars.items():
            # sharing of y-axis determines shape of 'key'
            _var, ecl = key if type(key) is tuple else (key, None)
            if _var == "E_GABA":
                continue
            labely = "Hz" if _var == "r" else "pA"
            tick_locs = _ax.yaxis.get_majorticklocs()
            sizey = len(tick_locs) > 1 and (tick_locs[1] - tick_locs[0])
            if int(f"{sizey:.0f}") == 0:
                fmt = "g"
            else:
                fmt = ".0f"
            use_scalebar(
                _ax,
                matchx=False,
                sizex=60,
                labelx="%s" % time_unit,
                labely=labely,
                fmt=fmt,
                loc="center right",
                bbox_to_anchor=(0, 0.5),
                textprops={"fontsize": "small"},
            )
        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self

    def plot_stats(
        self,
        drugs,
        e,
        ecl,
        fig,
        gs_stats,
        markersize,
        stats_data,
        stats_to_plot,
        x_axis,
        _ax_stats,
    ):
        import seaborn as sns

        ax_stats = []
        _ax_stat = None
        labels = []
        xlabel = f"{constants.G_GABA} modulation ($\\times $ 50 nS)"
        for stat_idx, stat_name in enumerate(stats_to_plot):
            if stat_name == "Number":
                marker = "o"
                label = "Number of Bursts\n(per min)"
            elif stat_name == "Amplitude":
                marker = "v"
                label = "Amplitude\n(Hz)"
            elif stat_name == "Duration":
                marker = "s"
                label = "Duration\n(%s)" % time_unit
            else:
                marker = "D"  # diamond
                label = "Interval\n(%s)" % time_unit
            ax_key = stat_name
            ax_y = _ax_stats[ax_key] if ax_key in _ax_stats else None
            _ax_stat = fig.add_subplot(gs_stats[stat_idx], sharex=_ax_stat, sharey=ax_y)
            if ax_key not in _ax_stats:
                _ax_stats[ax_key] = _ax_stat
            stat: pd.Series = stats_data[ecl, stat_name]
            df_stat = stat.explode().to_frame().reset_index()
            df_stat.columns = [xlabel, label]
            hues = [get_benzo_color(drug) for drug in drugs]
            hues = [
                benzo_color if ampa_color is None else ampa_color
                for benzo_color, ampa_color in hues
            ]
            sns.swarmplot(
                x=xlabel,
                y=label,
                hue=xlabel,
                palette=sns.blend_palette(hues, n_colors=len(hues)),
                data=df_stat,
                linewidth=0.1,
                marker=marker,
                zorder=-99,
                size=markersize,
                warn_thresh=0.1,
                ax=_ax_stat,
                clip_on=False,
            )
            means = [np.mean(data) for drug, data in stat.items()]
            _ax_stat.legend().remove()
            _ax_stat.plot(
                x_axis,
                means,
                "-",
                lw=1,
                marker="+",
                markersize=markersize * 2,
                color="k",
                zorder=99,
            )
            _ax_stat.set_ylabel(label)
            if stat_idx == len(stats_to_plot) - 1:
                _ax_stat.set_xlabel(xlabel)
            else:
                _ax_stat.set_xlabel("")
            _ax_stat.set_xticks(x_axis)
            labels = [get_drug_label(drug, 2) for drug in drugs]
            labels = [
                _benzo if _ampa is None else f"{_benzo}\n{_ampa}"
                for _benzo, _ampa in labels
            ]
            _ax_stat.set_xticklabels(labels)

            adjust_spines(_ax_stat, ["left"], position=5)
            ax_stats.append(_ax_stat)
        # adjust axes tick(label)s
        adjust_spines(_ax_stat, ["left", "bottom"], position=5, sharedx=True)
        return ax_stats

    def calc_stats_with_trace(
        self, _ax_trace, benzo_off_t, benzo_onset_t, drug, ecl, rates, stats_data
    ):
        benzo_color, ampa_color = get_benzo_color(drug)
        color = benzo_color if ampa_color is None else ampa_color
        burst_start_ts, burst_end_ts = burst_stats(
            rates,
            time_unit=time_unit,
            plot_fig=dict(
                ax=_ax_trace, lw=0.05, color=settings.COLOR.K, burst_kwargs=False
            ),
        )
        t_points, n_bursts = inst_burst_rate(
            burst_start_ts,
            rates.index[-1],
            window=60 * second,
            rolling=60 * second,
            time_unit=time_unit,
        )
        idx = 0
        for i, t in enumerate(t_points):
            if t >= benzo_onset_t:
                idx = i
                break
        t_points = t_points[idx:]
        n_bursts = n_bursts[idx:]
        # drug color above trace
        if drug >= 1 and type(drug) is float:
            _bs = int(drug)
            _ampa = np.round(drug - _bs, 2)
            logger.debug(f"drug: {drug}, _bs: {_bs}, _ampa: {_ampa}")

        if ampa_color is not None:
            _ax_trace.hlines(
                _ax_trace.get_ylim()[1] * 1.05,
                xmin=benzo_onset_t,
                xmax=benzo_off_t,
                lw=1,
                color=ampa_color,
                zorder=99,
                clip_on=False,
            )
            _ax_trace.hlines(
                _ax_trace.get_ylim()[1] * 1.01,
                xmin=benzo_onset_t * 0.8,
                xmax=benzo_off_t * 1.1,
                lw=1,
                color=benzo_color,
                zorder=99,
                clip_on=False,
            )
        else:
            _ax_trace.hlines(
                _ax_trace.get_ylim()[1] * 1.01,
                xmin=benzo_onset_t,
                xmax=benzo_off_t,
                lw=1,
                color=color,
                zorder=99,
                clip_on=False,
            )
        drug_label, ampa_label = get_drug_label(drug, 2)
        label = drug_label if ampa_label is None else ampa_label
        fontsize = "small" if ampa_label is None else "x-small"
        _ax_trace.annotate(
            label,
            xy=(0.5, 1),
            xycoords="axes fraction",
            c=color,
            va="bottom",
            ha="center",
            fontsize=fontsize,
        )
        if drug == 0:
            use_scalebar(
                _ax_trace,
                matchx=False,
                sizex=benzo_off_t - benzo_onset_t,
                labelx="%s" % time_unit,
                labely="Hz",
                fmt=".0f",
                loc="lower right",
                bbox_to_anchor=(0, -0.2),
                textprops={"fontsize": "small"},
            )
        else:
            adjust_spines(_ax_trace, [], 0, sharedx=True)
        # only consider when condition is active (from onset and until off)
        burst_start_ts = burst_start_ts[
            np.logical_and(
                burst_start_ts >= benzo_onset_t, burst_start_ts <= benzo_off_t
            )
        ]
        burst_end_ts = burst_end_ts[
            np.logical_and(burst_end_ts >= benzo_onset_t, burst_end_ts <= benzo_off_t)
        ]
        burst_durations, inter_burst_intervals = get_duration_interval(
            burst_start_ts, burst_end_ts
        )
        burst_xlims = list(zip(burst_start_ts, burst_end_ts))
        amps = []
        # for each burst, get the amplitude (height of burst to lowest rate)
        for xlim in burst_xlims:
            _x0, _x1 = xlim
            amp = np.max(rates[_x0:_x1]) - np.min(rates[_x0:_x1])
            amps.append(amp)
        stats_data.loc[drug, (ecl, "Amplitude")] = amps
        stats_data.loc[drug, (ecl, "Duration")] = burst_durations
        stats_data.loc[drug, (ecl, "Interval")] = inter_burst_intervals
        stats_data.loc[drug, (ecl, "Number")] = n_bursts

    def plot_vars(
        self,
        fig,
        _ax_vars,
        sub_gs_var,
        benzo_off_t,
        benzo_onset_t,
        drug,
        ecl,
        vars_to_plot,
        ax_egabas,
        split,
        sharey="ecl",
    ):
        # more elaborate plotting
        color, _ = get_benzo_color(drug)

        def plot_drugwash(drug_ax):
            # drug washing
            rect = Rectangle(
                xy=(benzo_onset_t, -1.0),
                width=benzo_off_t - benzo_onset_t,
                height=1,
                color=color,
            )
            drug_ax.add_artist(rect)
            drug_ax.set_ylim(-1, 1)  # include rectangle

            # add text on first column
            if self.E_Cl_0s.index(ecl) == 0:
                if drug > 1:
                    s = "benzodiazepine"
                elif drug < 1:
                    s = "picrotoxin"
                else:
                    s = f"{constants.G_GABA} = 50 nS"
                drug_ax.text(
                    benzo_off_t / 2 + benzo_onset_t / 2,
                    0,
                    s,
                    fontsize="x-small",
                    ha="center",
                    va="bottom",
                    c=color,
                )

        _gs_var = sub_gs_var[drug][ecl]

        for v, _var in enumerate(vars_to_plot):
            if sharey == "ecl":
                key = (_var, ecl)
            else:
                key = _var
            _ax = _ax_vars[key] if key in _ax_vars else None

            _ax = fig.add_subplot(
                _gs_var[v],
                sharex=_ax if not split else None,
                sharey=_ax if sharey else None,
            )
            if split and drug == 1 and _var == "E_GABA":
                plot_drugwash(_ax)
                adjust_spines(_ax, [])
                _ax.set_xlim(benzo_onset_t, benzo_off_t)
                # reassign
                _ax = ax_egabas[ecl]

            if key not in _ax_vars:
                _ax_vars[key] = _ax
            _var_df = self.df[(drug, ecl)].xs(f"{_var}_all", axis=1, level="var")
            # _var_df = self.df[(drug, ecl, f"{_var}_all")]
            if _var == "E_GABA":
                if not split or drug == 1:
                    if isinstance(_var_df, pd.DataFrame):
                        egaba = _var_df.iloc[0, 0]
                    else:
                        egaba = _var_df.iloc[0]
                    plot_state_colorbar(
                        self.df[drug, ecl, np.nan, 0],
                        "E_GABA_all",
                        fig=fig,
                        ax=_ax,
                        time_unit=time_unit / 10,
                        label_text="",
                    )
                    _ax.annotate(
                        f"{constants.EGABA} = {egaba:.1f} mV",
                        xy=(0, 1),
                        fontsize="x-small",
                        ha="left",
                        va="top",
                        path_effects=[
                            patheffects.withStroke(linewidth=1, foreground="w")
                        ],
                    )
                if drug != 1:
                    plot_drugwash(_ax)

            else:
                color = "k" if _var == "r" else settings.COLOR.inh
                _var_df.plot(ax=_ax, color=color, lw=0.25, alpha=0.6, legend=False)
                if _var.startswith("I"):
                    _ax.axhline(ls=":", lw=0.5, alpha=0.5, color="k", zorder=-99)
            if split:
                if drug == 1:
                    _ax.set_xlim(0, benzo_onset_t)
                else:
                    _ax.set_xlim(benzo_onset_t, benzo_off_t)
            else:
                _ax.set_xlim(0, _var_df.index[-1])
            adjust_spines(_ax, [], 0)


if __name__ == "__main__":
    ehco3 = -18
    phco3 = 0.2
    pcl = 1 - phco3
    mv_step = 2
    time_per_value = 60
    egabas = [-74, -60, -58, -46]
    E_Cl_0s = [round((e - phco3 * ehco3) / pcl, 2) for e in egabas]
    drugs = Drugs(benzo_strengths=(0, 0.25, 0.5, 1, 2, 3, 4, 5), E_Cl_0s=E_Cl_0s)
    drugs.run()
    drugs.plot()
    drugs.save_figure(use_args=False, close=False)
    plt.show()
