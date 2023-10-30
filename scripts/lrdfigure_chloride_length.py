from itertools import product
import time
from collections import OrderedDict, defaultdict
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cbook import flatten
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import settings
from core.analysis import burst_stats
from core.lrdfigure import MultiRunFigure, time_unit
from core.var_ranges import TAU_KCC2_LIST
from settings import logging
from style import text
from style.axes import (
    adjust_spines,
    colorbar_inset,
    colorline,
    letter_axes,
    use_scalebar,
)
from style.color import COLOR, lighten_color
from style.figure import new_gridspec
from style.plot_trace import plot_state_average
from utils.hashable import hashable

logger = logging.getLogger(__name__)


@hashable
class ChlorideLength(MultiRunFigure):
    fig_name = "figure_3b_chloride"

    monitors = {
        "r_all": True,
        "sp_all": False,
        "state_mon": ["E_GABA"],
        "synapse_mon": False,
    }
    ignore = []

    def __init__(
        self,
        tau_KCC2s=tuple(TAU_KCC2_LIST[::2]),
        E_Cl_0s=(-60,),
        g_GABAs=(50,),
        lengths=(15, 10, 7.5),
        **kwargs,
    ):
        super().__init__(
            OrderedDict(
                g_GABA_max={"range": g_GABAs, "title": text.G_GABA},
                tau_KCC2_E={"range": tau_KCC2s, "title": text.TAU_KCC2},
                tau_KCC2_I=text.TAU_KCC2,  # same as E
                E_Cl_0={"range": E_Cl_0s, "title": text.ECL0},
                length={"range": lengths, "title": "Length"},
            ),
            default_params=dict(dyn_cl=True),
            **kwargs,
        )

        self.tau_KCC2s = tau_KCC2s
        self.E_Cl_0s = E_Cl_0s
        self.g_GABAs = g_GABAs
        self.lengths = lengths

    def plot(
        self,
        timeit=True,
        burst_window=120,
        colorbar=False,
        histogram=True,
        plot_g_gabas=(50,),
        plot_tau_kcc2s=None,
        **kwargs,
    ):
        super().plot(**kwargs)
        logger.info("plotting")
        plot_time_start = time.time()

        gridspec = {
            "height_ratios": [2] * len(plot_g_gabas) + [1, 1],
        }

        self.tau_KCC2s = sorted(self.tau_KCC2s)
        self.g_GABAs = sorted(self.g_GABAs)
        self.E_Cl_0s = sorted(self.E_Cl_0s)
        self.lengths = sorted(self.lengths)

        if plot_tau_kcc2s is None:
            plot_tau_kcc2s = self.tau_KCC2s

        ncols = len(self.lengths)
        fig, gs = new_gridspec(
            nrows=len(gridspec["height_ratios"]),
            ncols=ncols,
            grid_kwargs=gridspec,
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL),
        )
        gs.update(top=0.9, bottom=0.1, left=0.1, right=0.93, hspace=0.35, wspace=0.3)

        if colorbar:
            axs = np.empty(shape=(len(plot_g_gabas) + 2, ncols), dtype=plt.Axes)
        else:
            axs = np.empty(shape=(len(plot_g_gabas) * 2 + 2, ncols), dtype=plt.Axes)

        ax_gaba: plt.Axes = None
        ax_r: plt.Axes = None

        self.fig, self.axs = fig, axs

        T = np.round(self.df.index.values[-1])
        bin_size = burst_window
        bins = np.arange(0, T + bin_size, bin_size)

        all_rates = self.df.xs(key="r_all", level="var", axis=1, drop_level=True)
        e_gaba_all = self.df.xs(key="E_GABA_all", level="var", axis=1, drop_level=True)
        e_gaba_E = self.df.xs(key="E_GABA_E", level="var", axis=1, drop_level=True)
        e_gaba_I = self.df.xs(key="E_GABA_I", level="var", axis=1, drop_level=True)
        t_index = self.df.index.values

        # min/max
        rmax = all_rates.values.max()
        vmin = e_gaba_all.values.min()
        vmax = e_gaba_all.values.max()
        egaba_norm = COLOR.EGABA_SM.norm
        egaba_cmap = COLOR.EGABA_SM.get_cmap()
        lighten = np.linspace(0.05, 1.1, len(self.lengths))

        lighten_g = np.linspace(0.8, 1.2, len(self.g_GABAs))
        cp = [settings.COLOR.TAU_SM.to_rgba(tau) for tau in self.tau_KCC2s]
        cs = [lighten_color(c, light) for c in cp for light in lighten_g]
        cs_arr = np.array(cs).reshape(len(self.tau_KCC2s), len(self.g_GABAs), 3)

        # data structures
        df_long = pd.DataFrame(
            columns=[
                "run_idx",
                "E_Cl",
                "g_GABA",
                "KCC2",
                "length",
                "Burst start time (s)",
                "Burst end time (s)",
            ]
        )
        df_final_egaba = pd.DataFrame(
            columns=[
                "run_idx",
                "E_Cl",
                "g_GABA",
                "KCC2",
                "length",
                "EGABA",
            ]
        )

        y_spacing = 20
        off_i = 0
        all_lines = []
        for len_idx, length in enumerate(self.lengths):
            bursts = []
            tau_bursts = defaultdict(
                list
            )  # create dict of list objects where key=tau, list = bursts
            all_lines = []
            for self_g_i, g_GABA in enumerate(self.g_GABAs):
                for (i, tau_KCC2), E_Cl_0, run_idx in product(
                    enumerate(self.tau_KCC2s), self.E_Cl_0s, range(len(self.seeds))
                ):
                    pop_rate = all_rates[g_GABA, tau_KCC2, E_Cl_0, length, run_idx]
                    burst_start_ts, burst_end_ts = burst_stats(
                        pop_rate,
                        rate_std_thresh=2,
                        time_unit=time_unit,
                        plot_fig=False,
                    )
                    logger.debug(f"burst_start_ts={burst_start_ts}")
                    bursts.append(burst_start_ts)
                    tau_bursts[tau_KCC2].append(burst_start_ts)
                    # store bursts
                    for start_t, end_t in zip(burst_start_ts, burst_end_ts):
                        df_long.loc[df_long.shape[0]] = [
                            run_idx,
                            E_Cl_0,
                            g_GABA,
                            tau_KCC2,
                            length,
                            start_t,
                            end_t,
                        ]
                    # egaba
                    final_egaba = e_gaba_all[
                        g_GABA, tau_KCC2, E_Cl_0, length, run_idx
                    ].values[-1]
                    df_final_egaba.loc[df_final_egaba.shape[0]] = [
                        run_idx,
                        E_Cl_0,
                        g_GABA,
                        tau_KCC2,
                        length,
                        final_egaba,
                    ]

                if g_GABA not in plot_g_gabas:
                    continue

                g_ax_index = plot_g_gabas.index(g_GABA)
                if colorbar:
                    axs[g_ax_index, len_idx] = ax_r = ax_gaba = fig.add_subplot(
                        gs[g_ax_index, len_idx], sharey=ax_r, sharex=ax_r
                    )
                else:
                    gs_g = GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=gs[g_ax_index, len_idx], hspace=0.05
                    )
                    axs[g_ax_index * 2, len_idx] = ax_gaba = fig.add_subplot(
                        gs_g[0], sharey=ax_gaba, sharex=ax_r
                    )
                    axs[g_ax_index * 2 + 1, len_idx] = ax_r = fig.add_subplot(
                        gs_g[1], sharey=ax_r, sharex=ax_r
                    )

                lines = []
                line_kwargs = dict(alpha=1, linewidth=0.5)
                d = {}

                # plot population rates
                for (i, tau_KCC2), E_Cl_0 in product(
                    enumerate(self.tau_KCC2s), self.E_Cl_0s
                ):
                    if tau_KCC2 not in plot_tau_kcc2s:
                        continue
                    pop_rate = all_rates[g_GABA, tau_KCC2, E_Cl_0, length, 0]
                    logger.debug(
                        f"plotting for g={g_GABA} tau={tau_KCC2} s with E_Cl_0={E_Cl_0} and length={length}"
                    )
                    egaba = e_gaba_all[g_GABA, tau_KCC2, E_Cl_0, length, 0].values
                    egaba_for_all_tau = e_gaba_all.loc[
                        :, pd.IndexSlice[g_GABA, :, E_Cl_0, length, 0]
                    ].values
                    min_egaba_for_all_tau = egaba_for_all_tau.min()
                    max_egaba_for_all_tau = egaba_for_all_tau.max()
                    d[f"{tau_KCC2}"] = egaba
                    if len_idx == 0:
                        ax_gaba.annotate(
                            f"{tau_KCC2:>4.1f}",
                            xy=(T, egaba[-1]),
                            xytext=(2, 0),
                            textcoords="offset points",
                            ha="left",
                            va="center_baseline",
                            fontsize="small",
                            c=cs_arr[i, self_g_i],
                        )
                    if colorbar:
                        ax_r.plot(
                            t_index,
                            pop_rate.values + y_spacing * (i + off_i),
                            c="w",
                            zorder=-i * 10 - 1,
                            alpha=1,
                            linewidth=1,
                        )
                        colorline(
                            t_index,
                            pop_rate.values + y_spacing * (i + off_i),
                            z=egaba,
                            cmap=egaba_cmap,
                            norm=Normalize(
                                vmin=min_egaba_for_all_tau, vmax=max_egaba_for_all_tau
                            ),
                            zorder=-i,
                            **line_kwargs,
                            ax=ax_r,
                        )
                        c = cp[i]
                    else:
                        c = cs_arr[i, self_g_i]
                        # offset = -r_all.iloc[-1] + d[f"t{tau_KCC2}"].iloc[-1]
                        ax_r.plot(
                            t_index,
                            pop_rate.values + y_spacing * (i + off_i),
                            c="w",
                            zorder=-i * 10 - 1,
                            alpha=1,
                            linewidth=1,
                        )
                        ax_r.plot(
                            t_index,
                            pop_rate.values + y_spacing * (i + off_i),
                            c=c,
                            zorder=-i * 10,
                            **line_kwargs,
                        )
                    lines.append(Line2D([], [], c=c, lw=3))
                if histogram:
                    burst_start_ts, burst_end_ts = burst_stats(
                        pop_rate,
                        rate_std_thresh=2,
                        time_unit=time_unit,
                        plot_fig=False,
                    )
                    logger.debug(f"burst_start_ts={burst_start_ts}")
                    bursts.append(burst_start_ts)
                    tau_bursts[tau_KCC2].append(burst_start_ts)
                    # store bursts
                    for start_t, end_t in zip(burst_start_ts, burst_end_ts):
                        df_long.loc[df_long.shape[0]] = [
                            run_idx,
                            E_Cl_0,
                            g_GABA,
                            tau_KCC2,
                            length,
                            start_t,
                            end_t,
                        ]

                ax_r.set_xlim(0, T)
                ax_r.set_xticks(np.arange(0, T + bin_size, bin_size))
                if g_ax_index == 0:
                    ax_gaba.set_title(
                        f"{length} ({text.MICROMETERS})", fontsize="medium"
                    )
                ax_r.set_ylim(0, rmax + y_spacing * (i + off_i))

                adjust_spines(ax_r, [], 0)

                # ax_r.grid(True, "major", "x", zorder=-99, linestyle='--')
                if not colorbar:
                    # plot example EGABAs
                    variables = list(d.keys())
                    d["t"] = t_index
                    state_mon = pd.DataFrame(d)
                    colors = cs_arr[:, self_g_i]
                    plot_state_average(
                        state_mon,
                        variables,
                        var_names=self.tau_KCC2s,
                        ax=ax_gaba,
                        alpha=1,
                        only_mean=True,
                        colors=colors,
                        lw=1,
                        linestyles="-",
                        time_unit=time_unit,
                        auto_legend=False,
                    )
                    # plot EGABA for E and I populations
                    # plot_state_average(pd.DataFrame(d_e), variables, var_names=self.tau_KCC2s, ax=ax_gaba,
                    #                    alpha=0.4, only_mean=False, colors=colors, lw=1.5, linestyles='-',
                    #                    time_unit=time_unit,
                    #                    auto_legend=False)
                    # plot_state_average(pd.DataFrame(d_i), variables, var_names=self.tau_KCC2s, ax=ax_gaba,
                    #                    alpha=0.4, only_mean=False, colors=colors, lw=1.5, linestyles='-',
                    #                    time_unit=time_unit,
                    #                    auto_legend=False)
                    ax_gaba.set_ylim(vmin, vmax)
                    ax_gaba.set_xlabel("")
                    ax_gaba.tick_params(axis="x", bottom=False)
                    adjust_spines(ax_gaba, ["left", "bottom"], 0)
                    ax_gaba.spines["bottom"].set_visible(False)
                    yticks = np.arange(np.round(-75), vmax, 5, dtype=int)
                    ax_gaba.set_yticks(yticks, minor=True)
                    ax_gaba.set_yticks(yticks[1::2])
                    ax_gaba.set_yticklabels(yticks[1::2])
                    # ax_gaba.grid(True, "major", "both", zorder=-99, linestyle='--')

                if len_idx == 0:
                    # scale bar
                    _r = (ax_r.get_ylim()[1]) / rmax  # scale based on y_spacing
                    sb = use_scalebar(
                        ax_r,
                        matchy=False,
                        sizey=10 * _r,
                        matchx=False,
                        sizex=0,
                        hidex=False,
                        hidey=False,
                        loc="center left",
                        bbox_to_anchor=(-0.1, 0.5),
                    )
                    sb.ylabel._text.set_fontsize("small")
                    sb.ylabel._text.set_rotation(90)
                    sb.ylabel.set_text("10 Hz")
                    sb = use_scalebar(
                        ax_r,
                        matchy=False,
                        sizey=0,
                        matchx=False,
                        sizex=100,
                        hidex=False,
                        hidey=False,
                        loc="upper left",
                        bbox_to_anchor=(0, 0.0),
                    )
                    sb.xlabel._text.set_fontsize("small")
                    sb.xlabel.set_text(f"{bin_size} s")
                    ax_r.yaxis.set_visible(True)
                    ax_r.set_yticks([])
                    ax_r.set_ylabel("population\nrate (Hz)")
                    ax_r.xaxis.set_ticks_position("none")  # remove ticks
                    # ax_r.grid(True, "major", "x", linestyle='--')
                    adjust_spines(ax_r, [], 0, sharedx=True, sharedy=True)
                    ax_gaba.set_ylabel(f"{text.EGABA}\n(mV)")
                    c = lighten_color(settings.COLOR.K, lighten_g[self_g_i])
                    c = settings.COLOR.G_GABA_SM.to_rgba(g_GABA)
                    ax_gaba.annotate(
                        f"{text.G_GABA}\n{g_GABA} nS",
                        xy=(-0.05, 1.05),
                        xycoords="axes fraction",
                        fontsize="medium",
                        ha="center",
                        va="bottom",
                        c=c,
                    )
                    c = lighten_color(settings.COLOR.K, lighten_g[self_g_i])
                    ax_gaba.annotate(
                        f"{text.TAU_KCC2} (s)",
                        xy=(1, 1),
                        xycoords="axes fraction",
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize="small",
                        c=c,
                    )

                elif len_idx == 1:
                    # ax_gaba.set_yticklabels([])
                    ax_gaba.set_ylabel("")

                if colorbar:
                    logger.debug(f"creating colorbar for E_Cl_0={E_Cl_0}")
                    cm = ScalarMappable(norm=egaba_norm, cmap=egaba_cmap)
                    from matplotlib.colorbar import Colorbar

                    cb: Colorbar = colorbar_inset(cm, ax=ax_r)
                    cb.set_label(ax_gaba.get_ylabel())
                    # ax_gaba.set_ylabel("")
                    # ax_gaba.set_ylim(cm.get_clim())
                    # ax_gaba.set_yticks([])
                all_lines += lines

            # histogram of bursts
            hist_bursts = []
            for tau_KCC2 in self.tau_KCC2s:
                hist_bursts += tau_bursts[tau_KCC2]
            # axs[-1, e].hist(hist_bursts, bins=np.arange(0, T + bin_size, bin_size),
            #                 align='mid', rwidth=1, color=cs)

        df_long["bin"] = pd.cut(
            df_long["Burst start time (s)"],
            bins=np.append(bins, bins[-1] + bin_size),
            labels=bins.astype(int),
        )
        num_bursts_col = f"Number of bursts\n(per {bin_size} s)"
        df_num_bursts = (
            df_long.groupby(["E_Cl", "g_GABA", "KCC2", "length", "bin", "run_idx"])
            .count()
            .reset_index()
            .rename(columns={"Burst start time (s)": num_bursts_col})
            .fillna(0)
        )
        df_num_bursts["KCC2 g_GABA"] = df_num_bursts.apply(
            lambda row: f"{row['KCC2']:.0f} {row['g_GABA']:.0f}",
            axis=1,
        )

        if ((un_kcc2 := df_num_bursts["KCC2"].unique()) == un_kcc2.astype(int)).all():
            df_num_bursts["KCC2"] = df_num_bursts["KCC2"].astype(int)
            df_final_egaba["KCC2"] = df_final_egaba["KCC2"].astype(int)

        hue_order = [g for g in self.g_GABAs]
        palette = [settings.COLOR.G_GABA_SM.to_rgba(g) for g in hue_order]
        long_bins = sorted(df_long["bin"].unique())
        share_lims = None
        sharey_bursts_lims = None
        sharey_egaba_lims = None
        for len_idx, length in enumerate(self.lengths):
            axs[-2, len_idx] = share_lims = sharey_egaba_lims = fig.add_subplot(
                gs[-2, len_idx], sharex=share_lims, sharey=sharey_egaba_lims
            )
            axs[-1, len_idx] = sharey_bursts_lims = fig.add_subplot(
                gs[-1, len_idx], sharex=share_lims, sharey=sharey_bursts_lims
            )

            sns.barplot(
                x="KCC2",
                y=num_bursts_col,
                hue="g_GABA",
                hue_order=hue_order,
                palette=palette,
                errorbar="se",
                errwidth=1,
                # capsize=0.05,
                zorder=5,  # one tail of errorbar
                data=df_num_bursts[
                    (df_num_bursts["length"] == length)
                    & (df_num_bursts["bin"] == long_bins[-1])
                ],
                ax=axs[-1, len_idx],
            )
            sns.violinplot(
                x="KCC2",
                y="EGABA",
                hue="g_GABA",
                hue_order=hue_order,
                palette=palette,
                dodge=True,
                fliersize=0.5,
                showfliers=False,
                linewidth=0.5,
                # showmeans=True,
                # meanprops=dict(color="k", mec="k", ms=2, marker="."),
                data=df_final_egaba[df_final_egaba["length"] == length],
                ax=axs[-2, len_idx],
            )

            shift = 0.5
            axs[-1, len_idx].set_xticks(
                np.arange(len(self.tau_KCC2s), dtype=int) - shift, minor=True
            )

            axs[-1, len_idx].set_yticks(
                np.arange(0, axs[-1, len_idx].get_ylim()[1], 2, dtype=int)
            )
            axs[-1, len_idx].set_yticks(
                np.arange(0, axs[-1, len_idx].get_ylim()[1], 1, dtype=int), minor=True
            )
            # get gap between major y ticks
            yticks = axs[-2, len_idx].get_yticks()
            y_gap = yticks[1] - yticks[0]
            minor_y_ticks_egaba = np.arange(yticks[0], yticks[-1], y_gap / 2).round(2)
            axs[-2, len_idx].set_yticks(minor_y_ticks_egaba, minor=True)

            # axs[-1, len_idx].grid(True, axis="y", which="both", alpha=0.4, zorder=-99, linestyle='--')
            # axs[-2, len_idx].grid(True, axis="y", which="both", alpha=0.4, zorder=-99, linestyle='--')
            axs[-1, len_idx].grid(True, axis="x", which="minor", alpha=0.4, zorder=-99, linestyle='--')
            axs[-2, len_idx].grid(True, axis="x", which="minor", alpha=0.4, zorder=-99, linestyle='--')
            # axs[-1, len_idx].set_xlim(-shift, len(bins) - 1 - shift)
            if len_idx == 0:
                axs[-1, len_idx].legend().remove()
                axs[-2, len_idx].legend(
                    title=text.G_GABA,
                    title_fontsize="small",
                    fontsize="small",
                    loc=(0, 1),
                    frameon=False,
                    ncol=len(self.g_GABAs),
                )
            else:
                axs[-1, len_idx].legend().remove()
                axs[-2, len_idx].legend().remove()
                axs[-1, len_idx].set_ylabel("")
                axs[-2, len_idx].set_ylabel("")

        axs[-1, 0].set_ylim(0)

        min_y_egaba, max_y_egaba = (
            df_final_egaba["EGABA"].min(),
            df_final_egaba["EGABA"].max(),
        )
        for ax_i in flatten(axs[-2, :]):
            ax_i.set_ylim(min_y_egaba, max_y_egaba)

        for ax_i in flatten(axs[: len(plot_g_gabas), :]):
            ax_i.set_xlim([0, T])
            ax_i.set_xticks(np.arange(0, T + bin_size, bin_size))
            ax_i.set_xticklabels([])
            # ax_i.grid(True, "major", "x", zorder=-99, linestyle='--')
            ax_i.tick_params(axis="x", bottom=False)

        for ax_i in flatten(axs[-2:, :]):
            ax_i.set_xlabel(text.TAU_KCC2)
            # ax_i.grid(True, "major", "both", zorder=-99, linestyle='--')
            ax_i.tick_params(axis="x", which="minor", bottom=False)

        for _ax in axs[:, 0]:
            letter_axes(
                _ax,
                xy=(0, 1),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
            )

        # set axes lines to be highest layer
        for ax_i in flatten(axs):
            ax_i.set_axisbelow(False)

        fig.align_labels(axs=list(flatten(axs)))

        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self

    def save_figure(self, **kwargs):
        figs = kwargs.pop("figs", None)
        if figs is None:
            figs = [self.fig]
        super().save_figure(figs=figs, **kwargs)

    def egaba_plot(self):
        fig, axs = plt.subplots(2)


if __name__ == "__main__":
    tau_KCC2_list = list(TAU_KCC2_LIST)

    ratio = tau_KCC2_list[1] / tau_KCC2_list[0]
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    # above ratio slightly off but results already cached.
    ratio = np.sqrt(2)
    tau_KCC2_list = tau_KCC2_list + [np.round(tau_KCC2_list[-1] * ratio, 1)]
    tau_KCC2_list = tau_KCC2_list + [np.round(tau_KCC2_list[-1] * ratio, 1)]

    print(ratio, tau_KCC2_list[::2])
    cl_length = ChlorideLength(
        tau_KCC2s=tau_KCC2_list[::2],
        g_GABAs=(50, 25, 100),
        lengths=(15, 10, 7.5, 6.25, 5),
        seeds=(
            None,
            1038,
            1337,
            1111,
            1234,
        ),
    )

    cl_length.run(duration=600)
    cl_length.plot(timeit=True, colorbar=False, plot_g_gabas=(50,))
    if settings.SAVE_FIGURES:
        cl_length.save_figure(use_args=False, close=False)
    plt.show()
