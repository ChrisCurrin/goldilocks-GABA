import copy
import itertools
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brian2 import Hz, second
from matplotlib import patheffects, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm, tqdm_pandas


import settings
from core.analysis import burst_stats, inst_burst_rate, spikes_to_rate
from core.lrdfigure import MultiRunFigure
from settings import constants, logging, time_unit
from style.axes import adjust_spines, colorbar_inset, use_scalebar
from style.figure import new_gridspec

logger = logging.getLogger(__name__)


class Tau(MultiRunFigure):
    fig_name = "figure_4_tau"

    monitors = {
        "r_all": True,
        "sp_all": False,
        "state_mon": ["E_GABA", "g_GABA"],
        "synapse_mon": False,
    }

    def __init__(
        self,
        tau_KCC2_E_list=settings.TAU_KCC2_LIST,
        tau_KCC2_I_list=settings.TAU_KCC2_LIST,
        g_GABA_list=settings.G_GABA_LIST,
        **kwargs,
    ):
        super().__init__(
            OrderedDict(
                g_GABA_max={"range": g_GABA_list, "title": constants.G_GABA},
                tau_KCC2_E={"range": tau_KCC2_E_list, "title": constants.TAU_KCC2_E},
                tau_KCC2_I={"range": tau_KCC2_I_list, "title": constants.TAU_KCC2_I},
            ),
            default_params=dict(E_Cl_0=-60, dyn_cl=True),
            **kwargs,
        )

        self.tau_KCC2_E_list = tau_KCC2_E_list
        self.tau_KCC2_I_list = tau_KCC2_I_list
        self.tau_KCC2_list = sorted(
            list(itertools.product(tau_KCC2_E_list, tau_KCC2_I_list))
        )
        self.tau_KCC2_list_columns = [f"{_t[0]}, {_t[1]}" for _t in self.tau_KCC2_list]
        self.g_GABA_list = g_GABA_list
        # self.iterables = [self.g_GABA_list, self.tau_KCC2_list_columns]
        # self.var_names = [constants.G_GABA, f'{constants.TAU_KCC2} (s) [PC, IN

    def run(self, subsample=100, pii0=False, **kwargs):
        super().run(subsample=subsample, **kwargs)

        if pii0:
            logger.info(f"PART 2: p_ii=0\n{'*'*20}")
            old_mon = copy.deepcopy(self.monitors)
            self.monitors = {
                **self.monitors,
                "sp_all": True,
                "synapse_mon": ["x_S", "u_S", "dI_S"],
            }

            tau_KCC2_E_list, tau_KCC2_I_list = (
                self.tau_KCC2_E_list,
                self.tau_KCC2_I_list,
            )

            # change ranges a bit
            mid_e = len(tau_KCC2_E_list) // 2
            mid_i = len(tau_KCC2_I_list) // 2
            self.iterables = [
                [50, self.g_GABA_list[-1]],
                [tau_KCC2_E_list[1], tau_KCC2_E_list[mid_e], tau_KCC2_E_list[-2]],
                [tau_KCC2_I_list[1], tau_KCC2_I_list[mid_i], tau_KCC2_I_list[-2]],
            ]
            self.df_main, self.df = self.df, None
            super().run(subsample=subsample, save_vars=True, p_ii=0.0001, **kwargs)
            self.df_ii0, self.df = self.df, self.df_main
            self.monitors = old_mon

        return self

    def process_data(self, burst_window=100 * second, **kwargs):
        df_long = pd.DataFrame(
            columns=[
                "run_idx",
                "g_GABA",
                "KCC2 E",
                "KCC2 I",
                "Burst start time (s)",
                "Burst end time (s)",
            ]
        )
        # get unique runs/seeds
        run_idxs = list(
            self.df.columns.levels[list(self.df.columns.names).index("run_idx")]
        )

        for g, g_GABA in enumerate(self.g_GABA_list):
            # get rate slice which has a series of (tau_e,tau_i) columns (and t as the index)
            df_r_all: pd.DataFrame = self.df[g_GABA].xs("r_all", axis=1, level="var")
            r_max = np.nanmax(df_r_all.values)
            # create bins
            bins = [
                f"{i*burst_window/time_unit:.0f}-{(i + 1)*burst_window/time_unit:.0f}"
                for i in range(round(df_r_all.index[-1] * time_unit / burst_window))
            ]

            # Data containers
            #   2D Dataframe with time bins as index and (tau_E, tau_I) pairs as columns
            df_bursts_bins = pd.DataFrame(
                columns=self.tau_KCC2_list_columns, index=bins
            )
            #   3D Dataframe with tau_I as index and (tau_E -> run_idx) as hierarchical columns
            df_taus = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [self.tau_KCC2_E_list, run_idxs],
                    names=[f"{constants.TAU_KCC2_E} (s)", "run index"],
                ),
                index=pd.Index(
                    self.tau_KCC2_I_list, name=f"{constants.TAU_KCC2_I} (s)"
                ),
            )

            for label, series in df_r_all.items():
                # extract taus from label
                tau_e, tau_i, run_idx = label
                taus = f"{tau_e:.0f}, {tau_i:.0f}"
                series.dropna(inplace=True)
                # calculate bursts per window, and plot if we've created an axis (plot_fig arg)
                start_ts, end_ts = burst_stats(
                    series,
                    rate_std_thresh=2.0,
                    rate_thresh=12 * Hz,
                    time_unit=time_unit,
                    plot_fig=False,
                )
                t_points, n_bursts = inst_burst_rate(
                    start_ts,
                    series.index.values[-1],
                    time_unit=time_unit,
                    window=burst_window,
                    rolling=burst_window,
                )
                # store bursts
                df_bursts_bins[taus] = n_bursts
                df_taus.loc[tau_i, (tau_e, run_idx)] = np.sum(n_bursts)
                for start_t, end_t in zip(start_ts, end_ts):
                    df_long.loc[df_long.shape[0]] = [
                        run_idx,
                        g_GABA,
                        tau_e,
                        tau_i,
                        start_t,
                        end_t,
                    ]
                if len(start_ts) == 0:
                    df_long.loc[df_long.shape[0]] = [
                        run_idx,
                        g_GABA,
                        tau_e,
                        tau_i,
                        np.nan,
                        np.nan,
                    ]
        df_num_bursts = (
            df_long.groupby(
                ["g_GABA", "KCC2 E", "KCC2 I", "run_idx"],
                as_index=False,
            )
            .count()
            .rename(columns={"Burst start time (s)": "Number of bursts"})
        )
        df_num_bursts[constants.TAU_KCC2_I] = df_num_bursts["KCC2 I"].astype(int)
        df_num_bursts[constants.TAU_KCC2_E] = df_num_bursts["KCC2 E"].astype(int)

        self.df_long = df_long
        self.df_bursts_bins = df_bursts_bins
        self.df_taus = df_taus
        self.df_num_bursts = df_num_bursts
        return (df_long, df_bursts_bins, df_taus, df_num_bursts)

    def plot_old(
        self,
        timeit=True,
        burst_window=100 * second,
        plot_g_GABA_list=None,
        rotation=45,
        **kwargs,
    ):
        super().plot(**kwargs)
        logger.info("plotting")
        # tau_KCC2_list = self.tau_KCC2_list
        tau_KCC2_list_columns = self.tau_KCC2_list_columns
        tau_KCC2_E_list, tau_KCC2_I_list = self.tau_KCC2_E_list, self.tau_KCC2_I_list
        if plot_g_GABA_list is None:
            plot_g_GABA_list = full_g_GABA_list = self.g_GABA_list
        else:
            full_g_GABA_list = self.g_GABA_list
        plot_time_start = time.time()

        # SETUP taus values to highlight and plot traces of
        mid_e = len(tau_KCC2_E_list) // 2
        mid_i = len(tau_KCC2_I_list) // 2
        plot_taus = list(
            itertools.product(
                [tau_KCC2_E_list[1], tau_KCC2_E_list[mid_e]],
                [tau_KCC2_I_list[1], tau_KCC2_I_list[mid_i]],
            )
        )
        plot_taus += list(
            itertools.product(
                [tau_KCC2_E_list[mid_e], tau_KCC2_E_list[-2]],
                [tau_KCC2_I_list[mid_i], tau_KCC2_I_list[-2]],
            )
        )
        plot_taus = sorted(set(plot_taus))

        # create a categorical colormap with 3x3 colors.
        cmap = settings.categorical_cmap(3, 4, "Dark2").colors
        cmap = np.delete(
            cmap, [2, 3, 7, 8, 11], axis=0
        )  # delete lightest row and gaps from plot_taus
        for i in range(len(cmap)):
            cmap[i] = settings.lighten_color(cmap[i], 1.2)

        cmap_tau_e_i = {
            (tau_e, tau_i): settings.lighten_color(
                settings.COLOR.TAU_PAL_DICT[tau_e],
                1.2 if tau_i > tau_e else 0.8 if tau_i < tau_e else 1,
            )
            for (tau_e, tau_i) in plot_taus
        }

        # create figure
        grid_kwargs = {
            "height_ratios": [0.6, 1, 2],
            "width_ratios": [1] * len(plot_g_GABA_list) + [1],
        }
        fig_bursts, gs = new_gridspec(
            len(grid_kwargs["height_ratios"]),
            len(grid_kwargs["width_ratios"]),
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL_no_cap),
            grid_kwargs=grid_kwargs,
        )
        gs.update(hspace=0.4, wspace=0.4, top=0.95, bottom=0.07, left=0.12, right=0.98)

        gs_bursts = GridSpecFromSubplotSpec(
            2,
            len(grid_kwargs["width_ratios"]),
            gs[-1, :],
            hspace=0.7,
            wspace=0.5,
            height_ratios=[1, 1],
            width_ratios=[1] * len(plot_g_GABA_list) + [1],
        )

        self.fig = fig_bursts

        # define some placeholders
        # init _ax to None for sharing y-axis among traces
        _ax: plt.Axes = None
        ax_bursts: plt.Axes = None
        ax_bursts_norm: plt.Axes = None
        gs_traces: GridSpecFromSubplotSpec = None
        df_long = pd.DataFrame(
            columns=[
                "run_idx",
                "g_GABA",
                "KCC2 E",
                "KCC2 I",
                "Burst start time (s)",
                "Burst end time (s)",
            ]
        )
        # get unique runs/seeds
        run_idxs = list(
            self.df.columns.levels[list(self.df.columns.names).index("run_idx")]
        )

        for g, g_GABA in enumerate(full_g_GABA_list):
            # get rate slice which has a series of (tau_e,tau_i) columns (and t as the index)
            df_r_all: pd.DataFrame = self.df[g_GABA].xs("r_all", axis=1, level="var")
            r_max = np.nanmax(df_r_all.values)
            # create bins
            bins = [
                f"{i*burst_window/time_unit:.0f}-{(i + 1)*burst_window/time_unit:.0f}"
                for i in range(round(df_r_all.index[-1] * time_unit / burst_window))
            ]

            # Data containers
            #   2D Dataframe with time bins as index and (tau_E, tau_I) pairs as columns
            df_bursts_bins = pd.DataFrame(columns=tau_KCC2_list_columns, index=bins)
            #   3D Dataframe with tau_I as index and (tau_E -> run_idx) as hierarchical columns
            df_taus = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [tau_KCC2_E_list, run_idxs],
                    names=[f"{constants.TAU_KCC2_E} (s)", "run index"],
                ),
                index=pd.Index(tau_KCC2_I_list, name=f"{constants.TAU_KCC2_I} (s)"),
            )
            # ticks based on time bins
            ticks = np.linspace(
                0,
                round(self.df.index[-1]),
                round(self.df.index[-1] / (burst_window / time_unit)) + 1,
            )

            # create axes and subgridspecs
            if g_GABA in plot_g_GABA_list:
                n = len(list(filter(lambda _x: _x[0] is not None, plot_taus)))
                gs_traces = GridSpecFromSubplotSpec(
                    n, 1, subplot_spec=gs[0, plot_g_GABA_list.index(g_GABA)], hspace=0.1
                )

            _ax_list = []

            for i, (label, series) in enumerate(df_r_all.items()):
                # extract taus from label
                tau_e, tau_i, run_idx = label
                taus = f"{tau_e:.0f}, {tau_i:.0f}"
                logger.debug(f"*** label={taus}")
                series.dropna(inplace=True)
                if (
                    (tau_e, tau_i) in plot_taus
                    and g_GABA in plot_g_GABA_list
                    and run_idx == 0
                ):
                    # pick out specific pair and first seed
                    # create axis and set kwargs
                    from matplotlib.axes import Axes

                    tau_idx = plot_taus.index((tau_e, tau_i))
                    _ax: Axes = fig_bursts.add_subplot(gs_traces[tau_idx], sharey=_ax)
                    plot_fig = {
                        "ax": _ax,
                        "lw": 0.05,
                        "color": cmap_tau_e_i[(tau_e, tau_i)],
                        "burst_kwargs": {},  # {"color": "k", "lw": 0.5, "ms": 0.01},
                    }
                    _ax_list.append(_ax)
                else:
                    # don't plot
                    _ax = None
                    tau_idx = 0
                    plot_fig = False

                # calculate bursts per window, and plot if we've created an axis (plot_fig arg)
                start_ts, end_ts = burst_stats(
                    series,
                    rate_std_thresh=2.0,
                    rate_thresh=12 * Hz,
                    time_unit=time_unit,
                    plot_fig=plot_fig,
                )
                t_points, n_bursts = inst_burst_rate(
                    start_ts,
                    series.index.values[-1],
                    time_unit=time_unit,
                    window=burst_window,
                    rolling=burst_window,
                )
                # store bursts
                df_bursts_bins[taus] = n_bursts
                df_taus.loc[tau_i, (tau_e, run_idx)] = np.sum(n_bursts)
                for start_t, end_t in zip(start_ts, end_ts):
                    df_long.loc[df_long.shape[0]] = [
                        run_idx,
                        g_GABA,
                        tau_e,
                        tau_i,
                        start_t,
                        end_t,
                    ]
                if len(start_ts) == 0:
                    df_long.loc[df_long.shape[0]] = [
                        run_idx,
                        g_GABA,
                        tau_e,
                        tau_i,
                        np.nan,
                        np.nan,
                    ]

                # plot this g gaba and taus combo
                if _ax is not None:
                    _ax.set_ylim(0, r_max * 1.5)
                    _ax.set_xlim(0, ticks[-1])
                    if tau_idx == 0:
                        # title
                        _ax.annotate(
                            f"{constants.TAU_KCC2}\nPC, IN",
                            xy=(-0.15, 1),
                            xycoords="axes fraction",
                            fontsize="x-small",
                            ha="center",
                            va="bottom",
                        )
                        _ax.set_title(
                            f"{constants.G_GABA} = {g_GABA} nS",
                            fontsize="small",
                            color=settings.COLOR.G_GABA_PAL_DICT[g_GABA],
                        )
                    if tau_idx == len(plot_taus) - 1 and g == len(plot_g_GABA_list) - 1:
                        # create scalebar representative of all traces
                        sb = use_scalebar(
                            _ax,
                            sizey=100,
                            matchx=False,
                            sizex=burst_window / second,
                            fmt=":.0f",
                            hidex=False,
                            hidey=False,
                            labely="Hz",
                            labelx="s",
                            loc="lower left",
                            bbox_to_anchor=(1, 0),
                        )
                        sb.ylabel._text.set_fontsize("x-small")
                        sb.xlabel._text.set_fontsize("x-small")
                    _ax.xaxis.set_ticks_position("none")  # remove tick visibility
                    _ax.set_xticks(ticks)  # set ticks for grid
                    _ax.grid(True, which="major", axis="x")
                    adjust_spines(_ax, [], 0, sharedx=True, sharedy=True)
                    _ax.set_ylabel(
                        f"{tau_e:>3.0f}, {tau_i:>3.0f}",
                        rotation=0,
                        fontsize="x-small",
                        color=cmap_tau_e_i[(tau_e, tau_i)],
                        ha="right",
                        va="center",
                    )
                    # disable and remove space for x axis
                    _ax.xaxis.set_visible(False)

                    # # annotate number of bursts
                    # sm = ScalarMappable(
                    #     norm=Normalize(-5, 10),
                    #     cmap=sns.light_palette(
                    #         cmap_tau_e_i[(tau_e, tau_i)],
                    #         n_colors=len(t_points),
                    #         reverse=False,
                    #         as_cmap=True,
                    #     ),
                    # )
                    # t0 = ticks[0]
                    # for t, n in zip(ticks[1:], n_bursts):
                    #     _ax.annotate(n, xy=(t/2 + t0/2, r_max), xytext=(0, 1), textcoords='offset points',
                    #                  fontsize='xx-small', va='center', ha='center',
                    #                  color=sm.to_rgba(n))
                    #     t0 = t

                    # _ax.annotate(f"{constants.TAU_KCC2}(PC) = {tau_e} s\n{constants.TAU_KCC2}(IN) = {tau_i} s",
                    #              xy=(0, 1), xycoords="axes fraction", ha="left", va="bottom")

            self.fig.align_ylabels(_ax_list)

            if g_GABA in plot_g_GABA_list:
                # aggregation figures

                # useful indices
                first_col = plot_g_GABA_list.index(g_GABA) == 0
                last_col = plot_g_GABA_list.index(g_GABA) == len(plot_g_GABA_list) - 1
                g_idx = plot_g_GABA_list.index(g_GABA)

                # add plots
                ax_bursts = fig_bursts.add_subplot(
                    gs_bursts[0, g_idx], sharey=ax_bursts
                )
                ax_bursts_norm = fig_bursts.add_subplot(
                    gs_bursts[1, g_idx], sharey=ax_bursts_norm
                )
                ax_heatmap = fig_bursts.add_subplot(gs[1, g_idx])

                # calculate mean and std
                df_taus = df_taus.infer_objects()
                # take mean of samples
                df_taus_mean = df_taus.T.groupby(level=0).mean().T
                df_taus_std = df_taus.T.groupby(level=0).std().T

                # norm values  (by first KCC2 I value)
                df_norms = 100 * df_taus / df_taus.iloc[0, :]
                df_norm = df_norms.T.groupby(level=0).mean().T
                df_norm_std = df_norms.T.groupby(level=0).std().T

                df_norm.loc[:, df_taus_mean.mean(axis=0) < 1] = np.nan
                df_norm_std.loc[:, df_taus_mean.mean(axis=0) < 1] = np.nan

                burst_plot_kwargs = dict(
                    marker="None",
                    capsize=0,
                    capthick=0.5,
                    lw=1,
                    legend="full" if first_col else False,
                    clip_on=False,
                    zorder=99,
                )
                burst_plot_kwargs["cmap"] = ListedColormap(
                    sns.color_palette("Purples_r", n_colors=len(tau_KCC2_I_list))
                )

                df_taus_mean.T.plot(ax=ax_bursts, yerr=df_taus_std, **burst_plot_kwargs)

                ax_bursts.set_xscale("log")
                ax_bursts.set_xticks(tau_KCC2_E_list)
                ax_bursts.set_xticks([], minor=True)
                ax_bursts.set_xticklabels(
                    tau_KCC2_E_list, rotation=rotation, ha="center", va="top"
                )
                ax_bursts.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0f"))
                ax_bursts.set_xlim(0)
                ax_bursts.grid(True, axis="y")
                if first_col:
                    ax_bursts.set_ylabel(
                        f"Number of bursts\n(in {ticks[-1]/second/60:.0f} minutes)"
                    )
                ax_bursts.yaxis.set_major_locator(MaxNLocator(4))
                ax_bursts.yaxis.set_minor_locator(MaxNLocator(8))
                ax_bursts.set_ylim(
                    0, max(df_taus_mean.values.max(), ax_bursts.get_ylim()[1])
                )
                ax_bursts.set_title(
                    f"{constants.G_GABA} = {g_GABA} nS",
                    fontsize="small",
                    color=settings.COLOR.G_GABA_PAL_DICT[g_GABA],
                    va="top",
                )

                # BARPLOT WITH ERR BANDS
                df_num_bursts = (
                    df_long[df_long["g_GABA"] == g_GABA]
                    .groupby(
                        ["KCC2 E", "KCC2 I", "run_idx"],
                        as_index=False,
                    )
                    .count()
                    .rename(columns={"Burst start time (s)": "Number of bursts"})
                )
                df_num_bursts[constants.TAU_KCC2_I] = df_num_bursts["KCC2 I"].astype(
                    int
                )
                df_num_bursts[constants.TAU_KCC2_E] = df_num_bursts["KCC2 E"].astype(
                    int
                )
                # sns.barplot(y='Number of bursts', x=constants.TAU_KCC2_E, hue=constants.TAU_KCC2_I,
                #              hue_order=tau_KCC2_I_list,
                #              errwidth=1, capsize=0.1,
                #              data=df_num_bursts, ax=ax_bursts, **burst_plot_kwargs)
                # df_norm.plot(ax=ax_bursts_norm, yerr=df_norm_std, **burst_plot_kwargs)

                # vertical barplot
                df_taus_mean.plot(
                    ax=ax_bursts_norm,
                    kind="bar",
                    stacked=True,
                    yerr=df_taus_std,
                    error_kw=dict(capsize=0, capthick=0.5, lw=0.5),
                    width=1,
                    cmap=LinearSegmentedColormap.from_list(
                        "TAU_cm", settings.COLOR.TAU_PAL
                    ),
                    legend=burst_plot_kwargs["legend"],
                    ylabel=f"Total\nnumber of bursts\n(in {ticks[-1]/second/60:.0f} minutes)",
                )
                # df_taus_mean.sum(axis=1).plot(
                #     ax=ax_bursts_norm,
                #     kind="line",
                #     yerr=df_taus_std.sum(axis=1),
                #     capsize=0,
                #     capthick=0.5,
                #     lw=2,
                #     color="k",
                #     zorder=101,
                # )

                # ax_bursts_norm.set_xscale('log')
                # ax_bursts_norm.set_xticks(tau_KCC2_I_list)
                # ax_bursts_norm.set_xticks([], minor=True)
                # ax_bursts_norm.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
                # ax_bursts_norm.set_xlim(0)

                ax_bursts_norm.set_title(
                    f"{constants.G_GABA} = {g_GABA} nS",
                    fontsize="small",
                    color=settings.COLOR.G_GABA_PAL_DICT[g_GABA],
                    va="top",
                )
                # ax_bursts_norm.yaxis.set_major_locator(MaxNLocator(4))
                # ax_bursts_norm.yaxis.set_minor_locator(MaxNLocator(8))
                # if first_col:
                #     ax_bursts_norm.set_ylabel(
                #         f"Total\nnumber of bursts\n(in {ticks[-1]/second/60:.0f} minutes)"
                #     )
                # # ax_bursts_norm.axhline(y=100, c='k', alpha=0.5, ls=':')
                # ax_bursts_norm.set_xticklabels(
                #     tau_KCC2_I_list, rotation=rotation, ha="center", va="top"
                # )
                self.fig.align_ylabels([ax_bursts, ax_bursts_norm])
                if first_col:
                    leg_kwargs = dict(
                        loc="center left",
                        bbox_to_anchor=(1.0, 0.5),
                        title_fontsize="x-small",
                        fontsize="x-small",
                        frameon=False,
                        borderaxespad=0.2,
                        borderpad=0,
                        handletextpad=1.5,
                        ncol=1,
                        columnspacing=-0.5,
                        reverse=True,
                        alignment="left",
                    )
                    ax_bursts.legend(
                        title=f"{constants.TAU_KCC2_I} (s)",
                        labelspacing=0.1,
                        **leg_kwargs,
                    )
                    ax_bursts_norm.legend(
                        title=f"{constants.TAU_KCC2_E} (s)",
                        labelspacing=0,
                        handlelength=1,
                        handleheight=1,
                        **leg_kwargs,
                    )

                vmax = ticks[-1] / 8
                vmin = 0.1
                # plot 0 values
                sns.heatmap(
                    df_taus_mean[::-1],
                    ax=ax_heatmap,
                    annot=False,
                    square=True,
                    mask=df_taus_mean[::-1] >= vmin,
                    lw=0.1,
                    cmap=sns.light_palette("lavender"),
                    cbar=False,
                )
                hm_cmap = "viridis"
                sns.heatmap(
                    df_taus_mean[::-1],
                    ax=ax_heatmap,
                    annot=False,
                    square=True,
                    # fmt='s',
                    # annot=df_taus_mean.apply(lambda s: s.map('${:>4.2f}'.format).str.cat(
                    #                              df_taus_std[s.name].map('${:3>.2f}'.format), " ± ")),
                    mask=df_taus_mean[::-1] < vmin,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=hm_cmap,
                    cbar=False,
                )
                ax_heatmap.set_xticks(
                    np.arange(len(tau_KCC2_E_list)) + 0.5, minor=False
                )
                ax_heatmap.set_xticklabels(
                    tau_KCC2_E_list, rotation=rotation, ha="center", va="top"
                )

                # annotate heatmap
                for idx, (tau_e, tau_i) in enumerate(plot_taus):
                    if tau_e is None:
                        continue
                    y = np.where(df_taus_mean[::-1].index == tau_i)[0]
                    x = np.where(df_taus_mean[::-1].columns == tau_e)[0]
                    rect = Rectangle(
                        (x, y),
                        1,
                        1,
                        fill=False,
                        edgecolor=cmap_tau_e_i[(tau_e, tau_i)],
                        lw=2,
                    )
                    rect.set_clip_on(False)
                    ax_heatmap.add_patch(rect)
                    ax_heatmap.annotate(
                        df_taus_mean.loc[tau_i, tau_e],
                        xy=(x + 0.5, y + 0.5),
                        color="w",
                        va="center",
                        ha="center",
                        fontsize="xx-small",
                        path_effects=[
                            patheffects.withSimplePatchShadow(offset=(0.5, -0.5))
                        ],
                    )

                if last_col:
                    # add colorbar without taking space
                    sm = ScalarMappable(
                        norm=Normalize(vmin=vmin, vmax=vmax), cmap=hm_cmap
                    )
                    cbar = colorbar_inset(
                        sm,
                        "outer right",
                        "10%",
                        ax=ax_heatmap,
                        inset_axes_kwargs=dict(borderpad=0.3),
                    )
                    cbar.set_label(
                        f"Number of bursts\n(in {ticks[-1]/second/60:.0f} minutes)"
                    )

        self.plot_g_gaba(ax_bursts, df_long, gs_bursts)

        plot_time = time.time()

        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self, df_long

    def plot(
        self,
        timeit=True,
        burst_window=100 * second,
        plot_ggaba=None,
        rotation=45,
        **kwargs,
    ):
        super().plot(**kwargs)
        logger.info("plotting")
        # tau_KCC2_list = self.tau_KCC2_list
        tau_KCC2_list_columns = self.tau_KCC2_list_columns
        tau_KCC2_E_list, tau_KCC2_I_list = self.tau_KCC2_E_list, self.tau_KCC2_I_list
        if plot_ggaba is None:
            plot_ggaba = full_g_GABA_list = self.g_GABA_list
        else:
            full_g_GABA_list = self.g_GABA_list
        plot_time_start = time.time()

        # SETUP taus values to highlight and plot traces of
        mid_e = len(tau_KCC2_E_list) // 2
        mid_i = len(tau_KCC2_I_list) // 2
        plot_taus = list(
            itertools.product(
                [tau_KCC2_E_list[1], tau_KCC2_E_list[mid_e]],
                [tau_KCC2_I_list[1], tau_KCC2_I_list[mid_i]],
            )
        )
        plot_taus += list(
            itertools.product(
                [tau_KCC2_E_list[mid_e], tau_KCC2_E_list[-2]],
                [tau_KCC2_I_list[mid_i], tau_KCC2_I_list[-2]],
            )
        )
        plot_taus = sorted(set(plot_taus))

        # create a categorical colormap with 3x3 colors.
        cmap = settings.categorical_cmap(3, 4, "Dark2").colors
        cmap = np.delete(
            cmap, [2, 3, 7, 8, 11], axis=0
        )  # delete lightest row and gaps from plot_taus
        for i in range(len(cmap)):
            cmap[i] = settings.lighten_color(cmap[i], 1.2)

        cmap_tau_e_i = {
            (tau_e, tau_i): settings.lighten_color(
                settings.COLOR.TAU_PAL_DICT[tau_e],
                1.2 if tau_i > tau_e else 0.8 if tau_i < tau_e else 1,
            )
            for (tau_e, tau_i) in plot_taus
        }

        ##############################
        # SETUP FIGURE
        # row 1: traces
        # row 2: num bursts (incl colorbar for heatmap) vs KCC2_E
        # row 3: heatmap of num bursts vs KCC2_E and KCC2_I, and hist of KCC2_I
        # row 4: colorbar for heatmap (horizontal)
        # each g_GABA gets 3 columns: colorbar, heatmap, histogram
        # final column is summaries
        ##############################
        # we use sum to flatten the list of lists that need to be repeated per g_GABA
        layout = [
            sum([[".", f"trace_ax_{g}", f"trace_ax_{g}"] for g in plot_ggaba], []),
            sum(
                [[f"heatmap_cbar_{g}", f"tau_KCC2_E_{g}", "."] for g in plot_ggaba],
                [],
            ),
            sum([[".", f"heatmap_{g}", f"tau_KCC2_I_{g}"] for g in plot_ggaba], []),
            sum([[".", ".", f"heatmap_hcbar_{g}"] for g in plot_ggaba], []),
        ]
        final_col = [["."], ["tau_KCC2_E_summary"], ["tau_KCC2_I_summary"], ["."]]

        for i in range(len(layout)):
            layout[i].extend(final_col[i])

        # ratios
        cbar_width = 0.05
        final_col_width = 1

        width_ratios = [cbar_width, 1, 1] * len(plot_ggaba) + [final_col_width]

        logger.debug(f"num cols: {len(layout[0])}")
        logger.debug(f"width_ratios: {len(width_ratios)}")

        # create figure
        fig_bursts, axes = plt.subplot_mosaic(
            layout,
            gridspec_kw={
                "height_ratios": [1, 0.5, 1, cbar_width],
                "width_ratios": width_ratios,
                "wspace": 0.05,
                "hspace": 0.05,
            },
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL),
        )
        gs_traces_g = {}
        for g_GABA in plot_ggaba:
            gs_traces_g[g_GABA] = axes[f"trace_ax_{g_GABA}"].get_gridspec()
            axes[f"trace_ax_{g_GABA}"].remove()

        ###############################
        # preprocess some useful vars
        ###############################
        (df_long, df_bursts_bins, df_taus, df_num_bursts) = self.process_data(
            burst_window=burst_window
        )
        vmax = df_num_bursts["Number of bursts"].max()

        # replace with a gridspec of len(plot_taus) rows. Filter out None values
        n = len(list(filter(lambda _x: _x[0] is not None, plot_taus)))
        # ticks based on time bins
        ticks = np.linspace(
            0,
            round(self.df.index[-1]),
            round(self.df.index[-1] / (burst_window / time_unit)) + 1,
        )
        # max rate value
        r_max = np.nanmax(self.df.xs("r_all", axis=1, level="var").values)

        ###############################
        # START plotting
        ###############################

        ###############################
        # RATE TRACES
        ###############################
        _ax = None
        for g_gaba in plot_ggaba:
            df_r_all: pd.DataFrame = self.df[g_GABA].xs("r_all", axis=1, level="var")
            gs_traces = GridSpecFromSubplotSpec(
                n, 1, subplot_spec=gs_traces_g[g_gaba], hspace=0.1
            )
            # store axes to align ylabels later
            _ax_list = []
            for i, (label, series) in enumerate(df_r_all.items()):
                # extract taus from label
                tau_e, tau_i, run_idx = label
                if not (tau_e, tau_i) in plot_taus or run_idx > 0:
                    continue
                tau_idx = plot_taus.index((tau_e, tau_i))
                taus = f"{tau_e:.0f}, {tau_i:.0f}"
                logger.debug(f"*** label={taus}")
                series.dropna(inplace=True)

                # pick out specific pair and first seed
                # create axis and set kwargs

                _ax: plt.Axes = fig_bursts.add_subplot(gs_traces[tau_idx], sharey=_ax)
                plot_fig = {
                    "ax": _ax,
                    "lw": 0.05,
                    "color": cmap_tau_e_i[(tau_e, tau_i)],
                    "burst_kwargs": {},  # {"color": "k", "lw": 0.5, "ms": 0.01},
                }
                _ax_list.append(_ax)

                # use existing plot functionality in burst_stats
                # (params should be same as in process_data)
                burst_stats(
                    series,
                    rate_std_thresh=2.0,
                    rate_thresh=12 * Hz,
                    time_unit=time_unit,
                    plot_fig=plot_fig,
                )

                _ax.set_ylim(0, r_max * 1.5)
                _ax.set_xlim(0, ticks[-1])
                if tau_idx == 0:
                    # title
                    _ax.annotate(
                        f"{constants.TAU_KCC2}\nPC, IN",
                        xy=(-0.15, 1),
                        xycoords="axes fraction",
                        fontsize="x-small",
                        ha="center",
                        va="bottom",
                    )
                    _ax.set_title(
                        f"{constants.G_GABA} = {g_GABA} nS",
                        fontsize="small",
                        color=settings.COLOR.G_GABA_PAL_DICT[g_GABA],
                    )
                if tau_idx == len(plot_taus) - 1 and g == len(plot_g_GABA_list) - 1:
                    # create scalebar representative of all traces
                    sb = use_scalebar(
                        _ax,
                        sizey=100,
                        matchx=False,
                        sizex=burst_window / second,
                        fmt=":.0f",
                        hidex=False,
                        hidey=False,
                        labely="Hz",
                        labelx="s",
                        loc="lower left",
                        bbox_to_anchor=(1, 0),
                    )
                    sb.ylabel._text.set_fontsize("x-small")
                    sb.xlabel._text.set_fontsize("x-small")
                _ax.xaxis.set_ticks_position("none")  # remove tick visibility
                _ax.set_xticks(ticks)  # set ticks for grid
                _ax.grid(True, which="major", axis="x")
                adjust_spines(_ax, [], 0, sharedx=True, sharedy=True)
                _ax.set_ylabel(
                    f"{tau_e:>3.0f}, {tau_i:>3.0f}",
                    rotation=0,
                    fontsize="x-small",
                    color=cmap_tau_e_i[(tau_e, tau_i)],
                    ha="right",
                    va="center",
                )
                # disable and remove space for x axis
                _ax.xaxis.set_visible(False)

        # align ylabels
        fig_bursts.align_ylabels(_ax_list)

        ############################
        # HEATMAPS of BURSTS
        ############################

        for g_gaba in plot_ggaba:
            ax_heatmap = axes[f"heatmap_{g_gaba}"]
            ax_tau_pc = axes[f"tau_KCC2_E_{g_gaba}"]
            ax_tau_in = axes[f"tau_KCC2_I_{g_gaba}"]
            ax_cbar = axes[f"heatmap_cbar_{g_gaba}"]
            ax_cbar_v = axes[f"heatmap_hcbar_{g_gaba}"]

            df = df_num_bursts[df_num_bursts["g_GABA"] == g_gaba]

            df[f"{constants.TAU_KCC2_E} (s)"] = df[constants.TAU_KCC2_E].apply(
                lambda x: f"{x:.0f}"
            )
            df[constants.TAU_KCC2_E] = (
                df[constants.TAU_KCC2_E].astype(int).astype("category")
            )
            df[constants.TAU_KCC2_I] = (
                df[constants.TAU_KCC2_I].astype(int).astype("category")
            )

            mean_num_bursts = (
                df.groupby(
                    ["g_GABA", constants.TAU_KCC2_E, constants.TAU_KCC2_I],
                )["Number of bursts"]
                .agg(["count", "sum", "mean", "std"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "Number of bursts",
                    }
                )
            )

            square_df = mean_num_bursts.pivot(
                index=constants.TAU_KCC2_I,
                columns=constants.TAU_KCC2_E,
                values="Number of bursts",
            )[::-1]
            sns.heatmap(
                square_df,
                ax=axes["heatmap"],
                cbar_ax=axes["heatmap_cbar"],
                # cbar_kws={"orientation": "horizontal"},
                cmap="viridis",
                # mask=square_df == 0,
                annot=False,
                fmt=".1f",
                annot_kws={"fontsize": 8},
                vmin=0.0,
                vmax=vmax,
            )

            # lineplot for tau_KCC2_E
            df[constants.TAU_KCC2_I] = df[constants.TAU_KCC2_I].astype(int)
            tau_i = sorted(df[constants.TAU_KCC2_I].unique())

            sns.lineplot(
                data=df,
                x=f"{constants.TAU_KCC2_E} (s)",
                y="Number of bursts",
                hue=constants.TAU_KCC2_I,
                hue_order=tau_i,
                palette="RdPu",
                ax=axes["tau_KCC2_E"],
                legend=True,
                err_style="bars",
            )
            axes["tau_KCC2_I"].set_xlim(0, vmax)
            axes["tau_KCC2_E"].set_ylim(0, vmax)
            axes["heatmap_cbar"].set_ylim(0, vmax)

            ratio = tau_i[1] / tau_i[0]
            bins = np.append(tau_i, df[constants.TAU_KCC2_I].max() * ratio)
            sns.histplot(
                data=df.groupby([constants.TAU_KCC2_I, constants.TAU_KCC2_E]).mean(
                    numeric_only=True
                ),
                y=constants.TAU_KCC2_I,
                weights="Number of bursts",
                stat="count",
                hue=constants.TAU_KCC2_E,
                hue_order=sorted(df[constants.TAU_KCC2_E].unique()),
                palette=COLOR.TAU_PAL,
                multiple="layer",
                element="step",
                bins=bins,
                ax=axes["tau_KCC2_I"],
            )
            # set x scale to log 2
            axes["tau_KCC2_I"].set_yscale("log", base=ratio)
            axes["tau_KCC2_I"].set_yticks(
                df[constants.TAU_KCC2_I].unique() + np.diff(bins) / 2
            )
            axes["tau_KCC2_I"].set_yticklabels(df[constants.TAU_KCC2_I].unique())
            axes["tau_KCC2_I"].set_ylim(bins[0], bins[-1])

            sns.despine(ax=axes["tau_KCC2_E"], bottom=True)
            axes["tau_KCC2_E"].set(xlabel="")
            axes["tau_KCC2_E"].tick_params(
                axis="x", which="both", bottom=True, top=False, labelbottom=False
            )
            axes["tau_KCC2_E"].set_yticks(np.arange(0, vmax + 1, 20))
            axes["tau_KCC2_E"].set_yticks(np.arange(0, vmax + 1, 10), minor=True)
            axes["tau_KCC2_E"].grid(
                axis="y", which="major", color="lightgrey", linestyle="--", zorder=-1
            )

            sns.despine(ax=axes["tau_KCC2_I"], left=True)
            axes["tau_KCC2_I"].set(ylabel="", xlabel="Number of bursts")
            axes["tau_KCC2_I"].tick_params(
                axis="y", which="both", left=True, labelleft=False
            )
            axes["tau_KCC2_I"].set_xticks(np.arange(0, vmax + 1, 20))
            axes["tau_KCC2_I"].set_xticks(np.arange(0, vmax + 1, 10), minor=True)
            axes["tau_KCC2_I"].grid(
                axis="x", which="major", color="lightgrey", linestyle="--", zorder=-99
            )

            sns.despine(ax=axes["heatmap_cbar"], left=True, bottom=True)
            axes["heatmap_cbar"].tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
                labelright=False,
            )
            # LEGEND
            handles = [
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fill=True,
                    color=COLOR.TAU_PAL_DICT[i],
                    alpha=0.5,
                    edgecolor=COLOR.TAU_PAL_DICT[i],
                )
                for i in sorted(df[constants.TAU_KCC2_E].unique())
            ]

            axes["tau_KCC2_I"].legend(
                handles,
                sorted(df[constants.TAU_KCC2_E].unique(), reverse=False),
                loc=(0, 1),
                ncol=len(df[constants.TAU_KCC2_E].unique()),
                handletextpad=0,
                handlelength=0,
                columnspacing=0.2,
                labelspacing=0,
                labelcolor=COLOR.TAU_PAL,
                fontsize="x-small",
                title=f"{constants.TAU_KCC2_E} (s)",
                title_fontsize="small",
                frameon=False,
            )
            leg = axes["tau_KCC2_E"].legend(
                tau_i[::-1],
                loc=(1, 0.05),
                handletextpad=0,
                handlelength=0,
                columnspacing=0.2,
                labelspacing=0,
                labelcolor=sns.color_palette("RdPu_r", len(tau_i)),
                fontsize="x-small",
                title=f"{constants.TAU_KCC2_I} (s)",
                title_fontsize="small",
                frameon=False,
            )

    def plot_ii0(
        self, df_run, fig_bursts, gs, plot_taus, plot_g_GABA_list, burst_window
    ):
        """No I to I connections"""
        from brian2 import defaultclock

        gs_egaba = GridSpecFromSubplotSpec(len(plot_taus), 1, gs[0, -1], hspace=0.1)
        ax_egaba = None
        for idx, (tau_e, tau_i) in enumerate(plot_taus):
            ax_egaba = fig_bursts.add_subplot(gs_egaba[idx], sharey=ax_egaba)
            df_run = self.df_ii0[plot_g_GABA_list[-1], tau_e, tau_i, 0]
            spk_mon = self.save_dests[(plot_g_GABA_list[-1], tau_e, tau_i, 0)]["sp_all"]
            df_run.t = df_run.index
            rate = spikes_to_rate(spk_mon, time_unit=time_unit)
            b_s_t, b_e_t = burst_stats(rate, time_unit=time_unit, plot_fig=False)
            bins, bursts = inst_burst_rate(
                b_s_t,
                rate.index.values[-1],
                time_unit=time_unit,
                window=burst_window,
                rolling=burst_window,
            )
            self.egaba_traces(
                ax_egaba,
                df_run,
                fig_bursts,
                idx == 0,
                int((df_run.index[1] - df_run.index[0]) / defaultclock.dt),
            )
            adjust_spines(ax_egaba, [], 0)
            ax_egaba.set_xlabel("")
            ax_egaba.set_ylabel("")

    def egaba_traces(self, _ax, df_run, fig_bursts, annotate, subsample_run_idx):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from style.plot_trace import plot_state_colorbar

        divider = make_axes_locatable(_ax)
        ax_egaba_e = divider.append_axes("top", size="40%", pad=0.0, sharex=_ax)
        ax_egaba_i = divider.append_axes("top", size="40%", pad=0.0, sharex=_ax)
        plot_state_colorbar(
            df_run,
            "E_GABA_E",
            fig=fig_bursts,
            ax=ax_egaba_e,
            time_unit=time_unit / subsample_run_idx,
            label_text=False,
            extent=None,
        )
        plot_state_colorbar(
            df_run,
            "E_GABA_I",
            fig=fig_bursts,
            ax=ax_egaba_i,
            time_unit=time_unit / subsample_run_idx,
            label_text=False,
            extent=None,
        )
        adjust_spines([ax_egaba_e, ax_egaba_i], [], 0)
        egaba_e0 = df_run.E_GABA_E.iloc[-1]
        egaba_e0_color = settings.COLOR.get_egaba_color(egaba_e0)
        egaba_i0 = df_run.E_GABA_I.iloc[-1]
        egaba_i0_color = settings.COLOR.get_egaba_color(egaba_i0)
        _ax.annotate(
            f"{egaba_e0:.2f}",
            xy=(1, 0.5),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            color=egaba_e0_color,
            fontsize=5,
        )
        _ax.annotate(
            f"{egaba_i0:.2f}",
            xy=(1, 0.5),
            xycoords="axes fraction",
            ha="left",
            va="top",
            color=egaba_i0_color,
            fontsize=5,
        )

        if annotate:
            # EGABA start
            egaba_e0 = df_run.E_GABA_E.iloc[0]
            egaba_e0_color = settings.COLOR.get_egaba_color(egaba_e0)
            egaba_i0 = df_run.E_GABA_I.iloc[0]
            egaba_i0_color = settings.COLOR.get_egaba_color(egaba_i0)
            _ax.annotate(
                f"{egaba_e0:.2f},",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(0, 10),
                textcoords="offset points",
                ha="right",
                va="center_baseline",
                color=egaba_e0_color,
                fontsize=5,
            )
            _ax.annotate(
                f" {egaba_i0:.2f}",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(0, 10),
                textcoords="offset points",
                ha="left",
                va="center_baseline",
                color=egaba_i0_color,
                fontsize=5,
            )
            _ax.annotate(
                "",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="center_baseline",
                fontsize=5,
                arrowprops=dict(arrowstyle="-|>", fc="k", ec="k"),
            )

    def plot_lines(self, plots=(("all", 10), ("E", 5), ("I", 5))):
        """
        Plot population rate, E_GABA, and g_GABA for each population in plots.
        Plots is a tuple of ('population)', <relative column width>)
        :param plots:
        :type plots: tuple[tuple]
        """
        unzip = list(zip(*plots))
        plot_cols = unzip[0]
        plot_widths = unzip[1]
        gridspec = {"height_ratios": [10] * 3, "width_ratios": plot_widths}

        fig, ax = plt.subplots(
            nrows=len(gridspec["height_ratios"]),
            ncols=len(gridspec["width_ratios"]),
            squeeze=False,
            gridspec_kw=gridspec,
            sharex="all",
            sharey="row",
            figsize=(settings.PAGE_H_FULL_no_cap, settings.PAGE_W_FULL),
        )
        self.fig, self.ax = fig, ax
        _ax = None
        fig.subplots_adjust(top=0.80, left=0.15, wspace=0.1)
        ax_gabas = ax[:, 0]

        tau_KCC2_E_list, tau_KCC2_I_list = self.tau_KCC2_E_list, self.tau_KCC2_I_list

        cmap = settings.categorical_cmap(
            tau_KCC2_E_list.__len__(), tau_KCC2_I_list.__len__()
        )
        p_i = 0

        _e_proportion = False
        norm = tau_KCC2_E_list[0]
        lw_norm = [tau_e / norm for tau_e in tau_KCC2_E_list]
        lw = np.linspace(0.5, 1, tau_KCC2_E_list.__len__())

        for col_name in self.df.columns.levels[1]:
            for p_i, p_nme in enumerate(plot_cols):
                if f"_{p_nme}" in col_name:
                    break
            if "E_GABA" in col_name:
                i = 0
            elif "g_GABA" in col_name:
                i = 1
            elif "r_" in col_name:
                i = 2
            else:
                raise IndexError(col_name)
            _ax = ax[i, p_i]
            df: pd.DataFrame = self.df.xs(col_name, axis=1, level=1)
            df.plot(
                ax=_ax,
                alpha=0.2,
                lw=0.2,
                legend=False,
                style=["-", "--", ":", "-."][: tau_KCC2_I_list.__len__()]
                * tau_KCC2_E_list.__len__(),
                cmap=cmap,
            )
            for l_idx, line in enumerate(_ax.get_lines()):
                e = l_idx // tau_KCC2_I_list.__len__()
                if i == 0:
                    line.set_linewidth(1)
                if _e_proportion:
                    line.set_linewidth(lw[e])
                    line.set_zorder(-lw_norm[e])

            ax_gabas[i].set_ylabel(
                col_name.replace("E_GABA", "EGABA\n(mV)")
                .replace("r_", "rates\n(Hz)_")
                .replace("g_GABA", "$g_{GABA}$\n(nS)")
                .replace("_E", " [E]")
                .replace("_I", " [I]")
                .replace("_all", " [$\mu$]"),  # noqa: W605
                rotation=90,
                va="center",
                ha="center",
                fontsize="x-small",
            )
        leg = ax_gabas[0].legend(
            loc=(0, 1.05),
            title="$\mathit{\\tau}$KCC2 [E,I] (seconds)",  # noqa: W605
            ncol=tau_KCC2_E_list.__len__(),
            fontsize="x-small",
            title_fontsize="small",
        )
        for line in leg.get_lines():
            line.set_linewidth(line.get_linewidth() * 2)
        adjust_spines(ax, ["left"], sharedx=True, position=0)
        # adjust_spines(ax[:-1, 1], [], position=0)
        ax[-2, 0].set_xbound(0, 0.99 * self.df.index[-1])
        fig.align_ylabels()
        # for ax_i in flatten(ax[:-1, :]):
        #     ax_i.xaxis.label.set_visible(False)
        #     ax_i.set_xticklabels([])
        for i in range(len(plot_widths)):
            ax[-1, i].set_xlabel(f"{constants.TIME}" + " (%s)" % time_unit)
        for i, col in enumerate(plot_cols):
            ax[0, i].set_title(col, fontsize="small", va="top")
        return _ax

    def create_sized_heatmap(self, df_num_bursts, tau_KCC2_E_list, tau_KCC2_I_list):
        fig, ax = plt.subplots()
        vmax = df_num_bursts["Number of bursts"].max()
        sns.scatterplot(
            x=constants.TAU_KCC2_E,
            y=constants.TAU_KCC2_I,
            hue="Number of bursts",
            size="Number of bursts",
            hue_norm=(0, vmax),
            size_norm=(0, vmax),
            sizes=(40, 100),
            marker="s",
            linewidth=0,
            ec="None",
            palette="mako_r",
            legend=False,
            alpha=0.4,
            data=df_num_bursts,
            ax=ax,
        )
        fig.colorbar(ax.scatter([], [], c=[], cmap="mako_r", vmin=0, vmax=vmax))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(tau_KCC2_E_list)
        ax.set_xticklabels(tau_KCC2_E_list)
        ax.set_yticks(tau_KCC2_I_list)
        ax.set_yticklabels(tau_KCC2_I_list)


if __name__ == "__main__":
    import argparse

    # process cmd args to take in tau_e index, tau_i index, g_gaba, seeds

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--tau_e_idx",
        type=int,
        nargs="*",
        default=[],
        help="Index of tau_e to run",
    )

    argparser.add_argument(
        "--tau_i_idx",
        type=int,
        nargs="*",
        default=[],
        help="Index of tau_i to run",
    )

    argparser.add_argument(
        "--g_gaba",
        type=int,
        nargs="*",
        default=[25, 50, 100, 200],
        help="GABA conductance to run",
    )

    argparser.add_argument(
        "--g_gaba_plot",
        type=int,
        nargs="*",
        default=[50, 200],
        help="GABA conductance to run",
    )

    args = argparser.parse_args()

    tau_KCC2_E_list = (
        [settings.TAU_KCC2_LIST[e_idx] for e_idx in args.tau_e_idx]
        if args.tau_e_idx
        else settings.TAU_KCC2_LIST
    )
    tau_KCC2_I_list = (
        [settings.TAU_KCC2_LIST[i_idx] for i_idx in args.tau_i_idx]
        if args.tau_i_idx
        else settings.TAU_KCC2_LIST
    )

    # process args as a string
    str_args = f"e_{tau_KCC2_E_list}_i_{tau_KCC2_I_list}_g_gaba_{args.g_gaba}".replace(
        " ", ""
    )
    __device_directory = f".cpp_{str_args}"

    print(str_args)
    print(f"{tau_KCC2_E_list=}")
    print(f"{tau_KCC2_I_list=}")
    print(f"{args.g_gaba=}")

    save_args = dict(use_args=False, close=True)

    plot_g_GABA_list = args.g_gaba_plot
    tau = Tau(
        tau_KCC2_E_list=tau_KCC2_E_list,
        tau_KCC2_I_list=tau_KCC2_I_list,
        g_GABA_list=args.g_gaba,
        seeds=(
            None,
            1038,
            1337,
            1111,
            1010,
        ),
        __device_directory=__device_directory,
    )

    tau.run(duration=600, nrn_idx_i=[0, 1, 2, 3])
    tau.plot(plot_g_GABA_list=plot_g_GABA_list)
    tau.save_figure(**save_args)
    plt.show()
