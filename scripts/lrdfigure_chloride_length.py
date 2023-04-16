import time
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brian2.units import second
from matplotlib.cbook import flatten
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import settings
from core.analysis import burst_stats
from core.lrdfigure import MultiRunFigure, time_unit
from settings import logging
from style import constants
from style.axes import (
    adjust_spines,
    colorbar_inset,
    colorline,
    letter_axes,
    use_scalebar,
)
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
        tau_KCC2s=(250, 100, 50, 30),
        E_Cl_0s=(-60,),
        g_GABAs=(50,),
        lengths=(15, 10, 7.5),
        **kwargs,
    ):
        super().__init__(
            OrderedDict(
                g_GABA_max={"range": g_GABAs, "title": constants.G_GABA},
                tau_KCC2_E={"range": tau_KCC2s, "title": constants.TAU_KCC2},
                tau_KCC2_I=constants.TAU_KCC2,  # same as E
                E_Cl_0={"range": E_Cl_0s, "title": constants.ECL0},
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
        burst_window=60 * second,
        colorbar=False,
        histogram=True,
        **kwargs,
    ):
        super().plot(**kwargs)
        logger.info("plotting")
        plot_time_start = time.time()

        gridspec = {
            "height_ratios": [0.5] * len(self.g_GABAs) + [1],
        }

        self.tau_KCC2s = sorted(self.tau_KCC2s)
        self.g_GABAs = sorted(self.g_GABAs)
        self.E_Cl_0s = sorted(self.E_Cl_0s)
        self.lengths = sorted(self.lengths)

        ncols = len(self.lengths)
        fig, gs = new_gridspec(
            nrows=gridspec["height_ratios"].__len__(),
            ncols=ncols,
            grid_kwargs=gridspec,
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL_no_cap),
        )
        gs.update(top=0.95, bottom=0.05, left=0.1, right=0.93, hspace=0.35, wspace=0.1)

        axs = np.empty(shape=(len(self.g_GABAs) * 2, ncols), dtype=plt.Axes)

        ax_gaba: plt.Axes = None
        ax_r: plt.Axes = None

        self.fig, self.axs = fig, axs

        T = np.round(self.df.index.values[-1])
        bin_size = 100
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
        norm = Normalize(vmin, vmax)
        lighten = np.linspace(0.05, 1.1, len(self.lengths))
        egaba_c = [
            settings.lighten_color(settings.COLOR.K, lighten[-1]),
            settings.lighten_color(settings.COLOR.K, lighten[0]),
        ]
        cmap = LinearSegmentedColormap.from_list("E_GABA_cm", egaba_c)
        lighten_g = np.linspace(0.6, 1.3, len(self.g_GABAs))
        cp = sns.color_palette("Set1", n_colors=len(self.tau_KCC2s))
        cs = [settings.lighten_color(c, light) for c in cp for light in lighten_g]
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

        y_spacing = 20
        off_i = 0
        all_lines = []
        for len_idx, length in enumerate(self.lengths):
            bursts = []
            tau_bursts = defaultdict(
                list
            )  # create dict of list objects where key=tau, list = bursts
            all_lines = []
            for g, g_GABA in enumerate(self.g_GABAs):
                gs_g = GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=gs[g, len_idx], hspace=0.05
                )
                axs[g * 2, len_idx] = ax_gaba = fig.add_subplot(
                    gs_g[0], sharey=ax_gaba, sharex=ax_r
                )
                axs[g * 2 + 1, len_idx] = ax_r = fig.add_subplot(
                    gs_g[1], sharey=ax_r, sharex=ax_r
                )

                lines = []
                line_kwargs = dict(alpha=1, linewidth=0.5)
                d = {}  # store egabas
                d_e = {}  # store egabas
                d_i = {}  # store egabas

                # plot population rates
                for i, tau_KCC2 in enumerate(self.tau_KCC2s):
                    for e, E_Cl_0 in enumerate(self.E_Cl_0s):
                        for run_idx in range(len(self.seeds)):
                            pop_rate = all_rates[
                                g_GABA, tau_KCC2, E_Cl_0, length, run_idx
                            ]
                            if run_idx == 0:
                                logger.debug(
                                    f"plotting for g={g_GABA} tau={tau_KCC2} s with E_Cl_0={E_Cl_0} and length={length}"
                                )
                                d[f"t{tau_KCC2}"] = e_gaba_all[
                                    g_GABA, tau_KCC2, E_Cl_0, length, run_idx
                                ]
                                d_e[f"t{tau_KCC2}"] = e_gaba_E[
                                    g_GABA, tau_KCC2, E_Cl_0, length, run_idx
                                ]
                                d_i[f"t{tau_KCC2}"] = e_gaba_I[
                                    g_GABA, tau_KCC2, E_Cl_0, length, run_idx
                                ]
                                egaba = d[f"t{tau_KCC2}"].values
                                if len_idx == 1:
                                    ax_gaba.annotate(
                                        f"{tau_KCC2:>4.0f}",
                                        xy=(T, egaba[-1]),
                                        xytext=(2, 0),
                                        textcoords="offset points",
                                        ha="left",
                                        va="center_baseline",
                                        fontsize="small",
                                        c=cs_arr[i, g],
                                    )
                                if colorbar:
                                    colorline(
                                        t_index,
                                        pop_rate.values,
                                        z=egaba,
                                        cmap=cmap,
                                        norm=norm,
                                        zorder=-i,
                                        **line_kwargs,
                                        ax=ax_r,
                                    )
                                    c = cp[i]
                                else:
                                    c = cs_arr[i, g]
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
                if g == 0 and len_idx == 0:
                    ax_r.set_xticks(np.arange(0, T + bin_size, bin_size))
                ax_r.set_ylim(0, rmax + y_spacing * (i + off_i))

                # adjust_spines(ax_r, ['left', 'bottom'], 0)
                # ax_r.spines['left'].set_alpha(0.25)
                # ax_r.spines['right'].set_linestyle(':')
                # if e == 0:
                #     ax_r.set_ylabel(f"{constants.G_GABA} = {g_GABA}\npopulation rate (Hz)")
                # else:
                #   ax_r.set_ylabel(f"population rate (Hz)")
                # ax_gaba = ax_r.twinx()
                # adjust_spines(ax_r, [], 0)

                ax_r.grid(True, "major", "x", zorder=-len(self.tau_KCC2s) * 10)

                # not the first plot
                adjust_spines(ax_r, [], 0)

                # plot example EGABAs
                variables = list(d.keys())
                d["t"] = t_index
                d_e["t"] = t_index
                d_i["t"] = t_index
                state_mon = pd.DataFrame(d)
                colors = cs_arr[:, g]
                plot_state_average(
                    state_mon,
                    variables,
                    var_names=self.tau_KCC2s,
                    ax=ax_gaba,
                    alpha=1,
                    only_mean=True,
                    colors=colors,
                    lw=2,
                    linestyles="--",
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
                yticks = np.arange(np.round(-75), vmax, 5)
                ax_gaba.set_yticks(yticks, minor=True)
                ax_gaba.set_yticks(yticks[1::2])
                ax_gaba.grid(True, "both", "both", zorder=-99)

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
                    ax_r.grid(True, "major", "x")
                    adjust_spines(ax_r, [], 0, sharedx=True, sharedy=True)
                    ax_gaba.set_ylabel(f"{constants.EGABA}\n(mV)")
                    c = settings.lighten_color(settings.COLOR.K, lighten_g[g])
                    ax_gaba.annotate(
                        f"{constants.G_GABA}\n{g_GABA} nS",
                        xy=(-0.05, 1.05),
                        xycoords="axes fraction",
                        fontsize="medium",
                        ha="center",
                        va="bottom",
                        c=c,
                    )

                elif len_idx == 1:
                    # ax_gaba.set_yticklabels([])
                    ax_gaba.set_ylabel("")
                    c = settings.lighten_color(settings.COLOR.K, lighten_g[g])
                    ax_gaba.annotate(
                        f"{constants.TAU_KCC2} (s)",
                        xy=(1, 1),
                        xycoords="axes fraction",
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize="small",
                        c=c,
                    )

                if colorbar:
                    logger.debug(f"creating colorbar for E_Cl_0={E_Cl_0}")
                    cm = ScalarMappable(norm=norm, cmap=cmap)
                    from matplotlib.colorbar import Colorbar

                    cb: Colorbar = colorbar_inset(cm, ax=ax_r)
                    cb.set_label(ax_gaba.get_ylabel())
                    ax_gaba.set_ylabel("")
                    ax_gaba.set_ylim(cm.get_clim())
                    ax_gaba.set_yticks([])
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

        share_lims = None
        for len_idx, length in enumerate(self.lengths):
            axs[-1, len_idx] = share_lims = fig.add_subplot(
                gs[-1, len_idx], sharex=share_lims, sharey=share_lims
            )
            sns.barplot(
                x="bin",
                y=num_bursts_col,
                hue="KCC2 g_GABA",
                hue_order=[
                    f"{k:.0f} {g:.0f}" for k in self.tau_KCC2s for g in self.g_GABAs
                ],
                palette=cs,
                # ci="sd",
                errwidth=0.2,
                capsize=0.05,
                zorder=5,  # one tail of errorbar
                data=df_num_bursts[df_num_bursts["length"] == length],
                ax=axs[-1, len_idx],
            )
            # sns.boxplot(x='bin', y=num_bursts_col, hue='KCC2 g_GABA',
            #             hue_order=[f"{k:.0f} {g:.0f}" for k in self.tau_KCC2s for g in self.g_GABAs],
            #             palette=cs,
            #             dodge=True,
            #             fliersize=0.5,
            #             showfliers=False,
            #             linewidth=0.2,
            #             showmeans=True,
            #             meanprops=dict(color='k', mec='k', ms=2, marker='.'),
            #             data=df_num_bursts[df_num_bursts['E_Cl'] == ecl],
            #             ax=axs[-1, e])

            shift = 0.5
            axs[-1, len_idx].set_xticks(np.arange(len(bins), dtype=int) - shift)
            axs[-1, len_idx].set_yticks(
                np.arange(0, axs[-1, len_idx].get_ylim()[1], 2, dtype=int)
            )
            axs[-1, len_idx].set_yticks(
                np.arange(0, axs[-1, len_idx].get_ylim()[1], 1, dtype=int), minor=True
            )
            axs[-1, len_idx].grid(
                True, axis="both", which="both", alpha=0.4, zorder=-99
            )
            axs[-1, len_idx].set_xlim(-shift, len(bins) - 1 - shift)
            if len_idx == 0:
                axs[-1, len_idx].legend().remove()
            else:
                axs[-1, len_idx].set_ylabel("")

        axs[-1, 0].set_ylim(0)

        tau_kcc2s_leg_v = [None] * len(self.tau_KCC2s) * (len(self.g_GABAs) - 1) + [
            (f"{tau_KCC2}") for tau_KCC2 in self.tau_KCC2s
        ]
        leg = axs[-1, 0].legend(
            all_lines,
            tau_kcc2s_leg_v,
            loc="upper left",
            bbox_to_anchor=(-0.0, 1),
            ncol=len(self.g_GABAs),
            columnspacing=-2.5,
            handlelength=0.1,
            handletextpad=3.8,
            labelspacing=0.3,
            borderaxespad=0.0,
            fontsize="small",
            frameon=True,
            facecolor="w",
            edgecolor="None",
            title=f"{constants.G_GABA} (nS)\n25  50 100  {constants.TAU_KCC2} (s)   ",
            title_fontsize="small",
        )

        for txt in leg.get_texts():
            txt.set_ha("right")

        # add block colours to bottom of plot
        all_lines_reorder = []
        for i in range(len(self.tau_KCC2s)):
            all_lines_reorder += all_lines[i :: len(self.tau_KCC2s)]  # noqa: E203
        # add white line to overlap last handle so they all appear equal
        all_lines_reorder.append(Line2D([], [], lw=3, c="w"))
        tau_kcc2s_leg_v = [None] * len(self.tau_KCC2s) * (len(self.g_GABAs) + 1)
        leg = axs[-1, -1].legend(
            all_lines_reorder,
            tau_kcc2s_leg_v,
            loc="upper left",
            bbox_to_anchor=(0.0, 0),
            ncol=len(tau_kcc2s_leg_v),
            columnspacing=0,
            handlelength=0.21,
            handletextpad=0,
            labelspacing=0,
            borderaxespad=0.0,
            borderpad=0.19,
            fontsize="small",
            frameon=False,
            facecolor="w",
            edgecolor="None",
        )

        for ax_i in flatten(axs[:-1, :]):
            ax_i.set_xlim([0, T])
            ax_i.set_xticklabels([])

        letter_axes(axs[::2, 0], xy=(-0.1, 1), ha="right", va="bottom")

        for ax_i in axs[-1, :]:
            ax_i.set_xlabel(f"{constants.TIME} bin" + " (%s)" % time_unit)
            # ax_i.set_xticks(list(range(0, T+bin_size, bin_size)))

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
    cl = ChlorideLength(seeds=(None,))
    cl.run(duration=600)
    cl.plot(timeit=True, colorbar=False)
    if settings.SAVE_FIGURES:
        cl.save_figure(use_args=False, close=False)
    plt.show()
