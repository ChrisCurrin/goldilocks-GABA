import itertools
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brian2 import ms
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from scipy import interpolate, stats

import settings
from core.analysis import burst_stats
from core.lrdfigure import MultiRunFigure, time_unit
from settings import logging
from style import constants
from style.axes import adjust_spines
from style.figure import new_gridspec
from style.text import math_fix

logger = logging.getLogger(__name__)
round_EGABA = f"{constants.E_GABA}"
sum_igaba = math_fix(
    r"$\sum_b^{bursts} \int_{b_{start} - 1 s}^{b_{start}} " f"{constants.I_GABA} (pA)$"
)
mean_igaba = math_fix(f"$\langle {constants.I_GABA} \\rangle (pA)$")  # noqa: W605


class Gve(MultiRunFigure):
    fig_name = "figure_5_gve"

    monitors = {
        "r_all": True,
        "sp_all": False,
        "state_mon": ["E_GABA", "g_GABA", "I_GABA_rec"],
        "synapse_mon": False,
    }

    # ignore = ['g_GABA_all', 'g_GABA_E', 'g_GABA_I']

    def __init__(self, gGABAs=(50.0,), tau_KCC2s=(100,), gGABAsvEGABA=None, **kwargs):
        self.gGABAs = gGABAs
        self.tau_KCC2s = tau_KCC2s
        self.gGABAsvEGABA = gGABAsvEGABA if gGABAsvEGABA is not None else gGABAs
        self.figs = []
        super().__init__(
            OrderedDict(g_GABA_max={"range": self.gGABAs, "title": constants.G_GABA}),
            **kwargs,
        )

    def run(
        self,
        subsample=10,
        time_per_value=60,
        EGABA_0=-74,
        EGABA_end=-40,
        mv_step=2,
        **kwargs,
    ):
        logger.info(f"PART 1\n{'*'*20}")
        self.time_per_value = time_per_value
        self.EGABA_0 = EGABA_0
        self.EGABA_end = EGABA_end
        self.mv_step = mv_step
        ehco3 = -18
        phco3 = 0.2
        pcl = 1 - phco3
        diff = EGABA_end - EGABA_0
        values = diff // mv_step
        self.num_ecl_steps = num_ecl_steps = values - 1
        duration = values * time_per_value
        ecl_0 = round((EGABA_0 - phco3 * ehco3) / pcl, 2)
        ecl_end = round((EGABA_end - phco3 * ehco3) / pcl, 2)
        kwargs["E_Cl_0"] = ecl_0
        kwargs["E_Cl_end"] = ecl_end

        manual_cl = kwargs.pop("manual_cl", True)
        super().__init__(
            OrderedDict(
                g_GABA_max={"range": self.gGABAsvEGABA, "title": constants.G_GABA},
            ),
            seeds=self.seeds,
            default_params=dict(
                manual_cl=manual_cl, duration=duration, num_ecl_steps=num_ecl_steps
            ),
            **kwargs,
        )
        super().run(subsample=subsample, **kwargs)

        self.df_main, self.df = self.df, None

        logger.info(f"PART 2 \n{'*'*20}")

        kwargs["E_Cl_0"] = ecl_0 / 2 + ecl_end / 2  # start at mid-point between the 2

        super().__init__(
            OrderedDict(
                g_GABA_max={"range": self.gGABAs, "title": constants.G_GABA},
                tau_KCC2_E={"range": self.tau_KCC2s, "title": constants.TAU_KCC2},
                tau_KCC2_I=constants.TAU_KCC2,
            ),
            seeds=self.seeds,
            default_params=dict(
                dyn_cl=True,
                duration=time_per_value * 10,
            ),
            **kwargs,
        )

        super().run(subsample=subsample, use_vaex=True, **kwargs)
        self.dynamic, self.df = self.df, self.df_main

        return self

    def process(self):
        run_idxs = list(range(len(self.seeds)))
        T = np.round(self.df.index.values[-1])
        bin_size = self.time_per_value
        bins = np.arange(0, T, self.time_per_value)
        _t_offset = int(time_unit / ms)

        ###############
        # static EGABA
        ###############
        df_g_E = pd.DataFrame(
            columns=[
                "run_idx",
                constants.I_GABA,
                constants.EGABA,
                constants.G_GABA,
                "Burst start time (s)",
            ]
        )
        if not isinstance(self.df, pd.DataFrame):
            self.df.rename("col_g__GABA__max___", "gGABA")

        for gGABA, run_idx in itertools.product(self.gGABAsvEGABA, run_idxs):
            if isinstance(self.df, pd.DataFrame):
                instance_df = self.df[gGABA, run_idx]
            else:
                instance_df = self.df[
                    (self.dynamic["gGABA"] == gGABA)
                    & (self.dynamic["run_idx"] == run_idx)
                ].to_pandas_df(["E_GABA_all", "I_GABA_all", "r_all"], index_name="Time")

            df_egaba = instance_df["E_GABA_all"]
            df_igaba = instance_df["I_GABA_all"]
            df_rates = instance_df["r_all"]
            burst_start_ts, burst_end_ts = burst_stats(
                df_rates, rate_std_thresh=3, time_unit=time_unit, plot_fig=False
            )
            logger.debug(
                f"gGABA={gGABA}, run_idx={run_idx} burst_start_ts={burst_start_ts}"
            )
            # store bursts
            for egaba, t_bin in zip(
                np.arange(self.EGABA_0, self.EGABA_end + self.mv_step, self.mv_step),
                bins,
            ):
                burst_ts = burst_start_ts[
                    (burst_start_ts >= t_bin) & (burst_start_ts < t_bin + bin_size)
                ]
                for start_t in burst_ts:
                    idx = np.argmin(np.abs(df_egaba.index - start_t))
                    egaba_t = np.round(df_egaba.iloc[idx], 2)
                    assert egaba == egaba_t
                    igaba_t_offset = idx - _t_offset
                    df_g_E.loc[df_g_E.shape[0]] = [
                        run_idx,
                        np.sum(df_igaba.iloc[igaba_t_offset:idx]),
                        egaba,
                        gGABA,
                        start_t,
                    ]
                if len(burst_ts) == 0:
                    # add expected observations

                    df_g_E.loc[df_g_E.shape[0]] = [
                        run_idx,
                        np.sum(df_igaba.iloc[-_t_offset:-1]),
                        egaba,
                        gGABA,
                        np.nan,
                    ]

        ###############
        # dynamic EGABA
        ###############
        df_g_tau = pd.DataFrame(
            columns=[
                "run_idx",
                constants.I_GABA,
                constants.EGABA,
                constants.G_GABA,
                constants.TAU_KCC2,
                "Burst start time (s)",
            ]
        )

        for gGABA, tau_KCC2, run_idx in itertools.product(
            self.gGABAs, self.tau_KCC2s, run_idxs
        ):
            if isinstance(self.dynamic, pd.DataFrame):
                instance_df = self.dynamic[gGABA, tau_KCC2, run_idx]
            else:
                instance_df = self.dynamic[
                    (self.dynamic[constants.G_GABA] == gGABA)
                    & (self.dynamic[constants.TAU_KCC2] == tau_KCC2)
                    & (self.dynamic["run_idx"] == run_idx)
                ].to_pandas_df(["E_GABA_all", "I_GABA_all", "r_all"], index_name="Time")

            df_egaba = instance_df["E_GABA_all"]
            df_igaba = instance_df["I_GABA_all"]
            df_rates = instance_df["r_all"]
            burst_start_ts, burst_end_ts = burst_stats(
                df_rates, rate_std_thresh=2, time_unit=time_unit, plot_fig=False
            )
            # only consider last bin (once network has reached steady-state
            burst_start_ts = burst_start_ts[burst_start_ts > self.time_per_value * 9]
            logger.debug(
                f"gGABA={gGABA}, tau_KCC2={tau_KCC2}, run_idx={run_idx} burst_start_ts={burst_start_ts}"
            )
            # store bursts
            for start_t in burst_start_ts:
                idx = np.argmin(np.abs(df_egaba.index - start_t))
                egaba_t = df_egaba.iloc[idx]
                igaba_t_offset = idx - _t_offset
                df_g_tau.loc[df_g_tau.shape[0]] = [
                    run_idx,
                    np.sum(df_igaba.iloc[igaba_t_offset:idx]),
                    egaba_t,
                    gGABA,
                    tau_KCC2,
                    start_t,
                ]
            if len(burst_start_ts) == 0:
                df_g_tau.loc[df_g_tau.shape[0]] = [
                    run_idx,
                    np.sum(df_igaba.iloc[-_t_offset:-1]),
                    df_egaba.iloc[-1],
                    gGABA,
                    tau_KCC2,
                    np.nan,
                ]

        # analysis

        df_g_E["bin"] = pd.cut(
            df_g_E["Burst start time (s)"],
            bins=np.append(bins, bins[-1] + bin_size),
            labels=bins.astype(int),
        )
        df_g_tau["bin"] = pd.cut(
            df_g_tau["Burst start time (s)"],
            bins=np.append(bins, bins[-1] + bin_size),
            labels=bins.astype(int),
        )
        num_bursts_col = f"Number of bursts\n(per {bin_size} s)"

        df_g_E_bursts = (
            df_g_E.groupby([constants.EGABA, constants.G_GABA, "bin", "run_idx"])
            .agg(["count", "sum", "mean"])
            .dropna()
            .drop(
                [
                    (constants.I_GABA, "count"),
                    ("Burst start time (s)", "sum"),
                    ("Burst start time (s)", "mean"),
                ],
                axis=1,
            )
        )
        # change from hierarchical index to flat index and change name
        df_g_E_bursts.columns = [sum_igaba, mean_igaba, num_bursts_col]
        df_g_E_bursts.reset_index(inplace=True)
        df_g_E_bursts["hyperpolaring\nEGABA"] = pd.cut(
            df_g_E_bursts[constants.EGABA], bins=[-100, -60, 0], labels=[True, False]
        )

        df_g_tau[round_EGABA] = df_g_tau[constants.EGABA].round(0)

        df_g_tau_bursts = (
            df_g_tau.groupby(
                [round_EGABA, constants.G_GABA, constants.TAU_KCC2, "bin", "run_idx"]
            )
            .agg(["count", "sum", "mean"])
            .dropna()
            .drop(
                [
                    (constants.I_GABA, "count"),
                    constants.EGABA,
                    ("Burst start time (s)", "sum"),
                    ("Burst start time (s)", "mean"),
                ],
                axis=1,
            )
        )
        # change from hierarchical index to flat index and change name
        df_g_tau_bursts.columns = [sum_igaba, mean_igaba, num_bursts_col]
        df_g_tau_bursts.reset_index(inplace=True)
        # df_mean = df_g_tau.groupby([round_EGABA, constants.G_GABA, constants.TAU_KCC2, 'bin', 'run_idx'],
        #                            as_index=False).mean().dropna()

        self.df_g_E = df_g_E
        self.df_g_E_bursts = df_g_E_bursts
        self.df_g_tau = df_g_tau
        self.df_g_tau_bursts = df_g_tau_bursts
        self.num_bursts_col = num_bursts_col
        self.sum_igaba = sum_igaba
        self.mean_igaba = mean_igaba

    def plot(self, timeit=True, **kwargs):
        super().plot(**kwargs)
        logger.info("plotting")
        plot_time_start = time.time()

        fig, gs = new_gridspec(
            2,
            4,
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_half),
            grid_kwargs=dict(
                width_ratios=(1, 0.08, 1, 0.05),
                height_ratios=(0.5, 1),
                wspace=1,
                hspace=0.4,
                left=0.1,
                right=0.95,
                bottom=0.2,
                top=0.95,
            ),
        )
        fig, axes = plt.subplot_mosaic(
            [
                ["static_egaba", "static_egaba_cax", "."],
                [".", ".", "i_gaba_cax"],
                ["tau_kcc2", ".", "i_gaba"],
            ],
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_half),
            gridspec_kw=dict(
                width_ratios=(1.5, 0.05, 1),
                height_ratios=(0.5, 0.1, 1),
                wspace=0.5,
                hspace=0.5,
                left=0.1,
                right=0.93,
                bottom=0.15,
                top=0.95,
            ),
        )

        self.figs.append(fig)
        # static_ax = fig.add_subplot(gs[0, 0])
        static_ax = axes["static_egaba"]
        self.plot_staticegaba(ax=static_ax)
        static_ax.set_xlim(10, xmax=1000)
        static_ax.set_ylim(0, ymax=13)
        static_ax.set_xlabel(f"{constants.G_GABA} (nS)")
        # sns.scatterplot(x=constants.G_GABA, y=round_EGABA,
        #                 hue=constants.TAU_KCC2,
        #                 size=self.num_bursts_col,
        #                 # palette="coolwarm",
        #                 # ec='None',
        #                 # marker='s',
        #                 data=self.df_g_tau_bursts,
        #                 ax=ax[1])
        fig_size = fig.get_size_inches()
        # tau_ax = fig.add_subplot(gs[1, :2])
        tau_ax = axes["tau_kcc2"]
        # cax = fig.add_subplot(gs[0, 1])
        cax = axes["static_egaba_cax"]
        self.plot_taukcc2(
            ax=tau_ax,
            cax=cax,
            min_s=fig_size[0] * fig_size[1],
        )
        tau_ax.set_xlabel(f"{constants.G_GABA} (nS)")

        # igaba_ax = fig.add_subplot(gs[1, -2])
        # cax = fig.add_subplot(gs[1, -1])
        igaba_ax = axes["i_gaba"]
        igaba_cax = axes["i_gaba_cax"]
        self.plot_igaba(fig=fig, ax=igaba_ax, cax=igaba_cax)

        # static_ax.set_title(constants.STATIC_CHLORIDE_STR_LONG, fontsize='medium', va='top', ha='center')
        # tau_ax.set_title(constants.DYNAMIC_CHLORIDE_STR_ABBR, fontsize='medium', va='center', ha='center')

        self.plot_heatmap()
        self.jointplot()
        self.i_explore()

        # fig.tight_layout()

        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self

    def plot_taukcc2(
        self, ax=None, cax=None, min_s=None, plot_3d=False, norm=None, cmap=None
    ):
        if ax is None:
            fig, ax_g_v_tau = plt.subplots(figsize=(6, 4))
            self.figs.append(fig)
        else:
            fig = ax.figure
            ax_g_v_tau = ax
        if min_s is None:
            fig_size = fig.get_size_inches()
            min_s = fig_size[0] * fig_size[1]
        if norm is None:
            norm = settings.COLOR.EGABA_SM.norm
        if cmap is None:
            cmap = settings.COLOR.EGABA_SM.cmap
            mesh_cmap = ListedColormap(
                sns.color_palette("Blues_r", self.num_ecl_steps * 2)
            )
        else:
            mesh_cmap = cmap
        p = self.df_g_tau.pivot_table(
            values=constants.EGABA, index=constants.TAU_KCC2, columns=constants.G_GABA
        )
        x = p.columns
        y = p.index

        # smoothing of EGABA for cleaner plot
        # f = interpolate.interp2d(x, y, p, kind="cubic")
        # use newer scipy version as interp2d is deprecated
        f = interpolate.RectBivariateSpline(x, y, p.T)

        new_x = np.round(np.arange(x[0], x[-1] + 1e-5, (x[1] - x[0]) / 10), 5)
        new_y = np.round(np.arange(y[0], y[-1] + 1e-5, (y[1] - y[0]) / 10), 5)
        XX, YY = np.meshgrid(new_x, new_y)

        ZZ = f(new_x, new_y)
        df_zz = pd.DataFrame(ZZ).T
        eg_df = np.abs(df_zz - -60)
        eg_idx = eg_df.idxmin(axis=0)
        eg_val = eg_df.min(axis=0)
        eg_idx = eg_idx[eg_val < 0.8]
        min_val = (
            new_x[df_zz.min().argmin()],
            new_y[df_zz.min(axis=1).argmin()],
            ZZ.min(),
        )
        max_val = (
            new_x[df_zz.max().argmax()],
            new_y[df_zz.max(axis=1).argmax()],
            ZZ.max(),
        )

        s = (
            pd.Series(new_y[eg_idx], index=new_x[eg_idx.index])
            .rolling(window=20, win_type="triang", center=True)
            .mean()
        )
        if plot_3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            n = self.df_g_tau_bursts.pivot_table(
                values=self.num_bursts_col,
                index=constants.TAU_KCC2,
                columns=constants.G_GABA,
            )
            xx, yy = np.meshgrid(x, y)
            ax.scatter(
                np.log10(xx),
                np.log10(yy),
                p + 1,
                s=n * 20,
                c=p.values.ravel(),
                cmap=cmap,
                norm=norm,
                depthshade=False,
                ec="k",
            )
            mesh = ax.plot_surface(
                np.log10(XX),
                np.log10(YY),
                ZZ,
                rcount=200,
                ccount=200,
                cmap="coolwarm",
                norm=norm,
                zorder=-99,
            )
            df_zz[eg_df > 0.1] = np.nan
            ax.scatter(
                np.log10(XX),
                np.log10(YY),
                df_zz.values,
                s=1,
                c="k",
                depthshade=False,
            )
            ax.set_ylim(np.log10(1), np.log10(1000))
            ax.set_xlim(np.log10(50), np.log10(1600))
            # ax.view_init(elev=10., azim=-45)
        else:
            mesh = ax_g_v_tau.pcolormesh(
                XX, YY, ZZ.T, cmap=mesh_cmap, norm=norm, zorder=-99
            )
            # plot EGABA = -60 mV line
            ax_g_v_tau.plot(s, c="k", ls="--", zorder=1)
            sizes = (min_s / 2, 10 * min_s)

            # get average number of bursts per g_gaba, tau_kcc2
            mean_bursts_per_g_gaba_tau_kcc2 = self.df_g_tau_bursts.groupby(
                [constants.G_GABA, constants.TAU_KCC2], as_index=False
            ).mean(numeric_only=True)
            mean_bursts_per_g_gaba_tau_kcc2[
                self.num_bursts_col
            ] = mean_bursts_per_g_gaba_tau_kcc2[self.num_bursts_col].round(1)
            # plot EGABA as hue, but use colorbar as legend
            sns.scatterplot(
                y=constants.TAU_KCC2,
                x=constants.G_GABA,
                hue=round_EGABA,
                hue_norm=norm,
                size=self.num_bursts_col,
                sizes=sizes,
                palette=cmap,
                marker="o",
                data=mean_bursts_per_g_gaba_tau_kcc2,
                legend="full",
                ec="w",
                ax=ax_g_v_tau,
                zorder=99,
                clip_on=False,
            )
            handles, labels = ax_g_v_tau.get_legend_handles_labels()
            # only include sizes (exclude egaba)
            idx_bursts = labels.index(self.num_bursts_col) + 1
            handles, labels = handles[idx_bursts:], labels[idx_bursts:]

            # choose first, last and some middle labels
            labels = [
                labels[0],
                labels[round(len(labels) * 1 / 4)],
                labels[round(len(labels) * 1 / 2)],
                labels[round(len(labels) * 3 / 4)],
                labels[-1],
            ]
            handles = [
                handles[0],
                handles[round(len(handles) * 1 / 4)],
                handles[round(len(handles) * 1 / 2)],
                handles[round(len(handles) * 3 / 4)],
                handles[-1],
            ]

            leg = ax_g_v_tau.legend(
                handles[::-1],  # reverse
                labels[::-1],
                loc="lower left",
                bbox_to_anchor=(1, 0),
                ncol=1,
                fontsize="x-small",
                labelspacing=1,
                frameon=False,
                title=self.num_bursts_col,
                title_fontsize="x-small",
            )
            # overlay lines
            for line, label in zip(leg.legend_handles, leg.texts):
                # line.set_position((0, 0))
                line.set_clip_on(False)
                line.set_zorder(99)
                line.set_edgecolor("w")

            ax_g_v_tau.set_yscale("log")
            tau_KCC2_list = self.df_g_tau[constants.TAU_KCC2].unique()
            ax_g_v_tau.set_yticks(tau_KCC2_list)
            ax_g_v_tau.set_yticks([], minor=True)
            ax_g_v_tau.set_yticklabels(
                [f"{t:.0f}" if t > 10 else f"{t:.1f}" for t in tau_KCC2_list]
            )
            # ax_g_v_tau.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

            ax_g_v_tau.set_ylim(tau_KCC2_list[0])
            g_gaba_list = self.df_g_tau[constants.G_GABA].unique()
            ax_g_v_tau.set_xlim(g_gaba_list[0])
            if g_gaba_list[-1] > 100:
                ax_g_v_tau.set_xscale("log")
                ax_g_v_tau.set_xlim(g_gaba_list[0], g_gaba_list[-1])
            annot_kws = dict(
                fontsize="xx-small", c="k", ha="center", va="center", zorder=101
            )
            ax_g_v_tau.annotate(f"{min_val[-1]:.1f}", xy=min_val[:-1], **annot_kws)
            ax_g_v_tau.annotate(f"{max_val[-1]:.1f}", xy=max_val[:-1], **annot_kws)
            sns.despine(ax=ax_g_v_tau, offset=5, trim=True)
        if cax is None:
            cbar = fig.colorbar(mesh, ax=ax_g_v_tau, cmap=cmap, norm=norm)
        else:
            cbar = fig.colorbar(
                mesh,
                cax=cax,
                cmap=cmap,
                norm=norm,
            )
        cbar.set_label(f"{constants.EGABA} (mV)")
        cbar.outline.set_visible(False)
        # cbar.set_ticks(np.arange(norm.vmin, norm.vmax+1, 10))
        cbar.minorticks_on()

    def i_explore(self, i_metric: str = None):
        if i_metric is None:
            i_metric = self.sum_igaba
        fig, ax = plt.subplots(nrows=3, sharey=True, sharex=True)
        data = self.df_g_tau_bursts[
            (self.df_g_tau_bursts[i_metric] < 1e6)
            & (self.df_g_tau_bursts[i_metric] > -1e6)
        ]
        logger.debug(f"len(data) = {len(data)}")

        p = self.df_g_tau_bursts.pivot_table(
            values=self.num_bursts_col,
            columns=constants.E_GABA,
            index=constants.G_GABA,
        )
        x = p.columns
        y = p.index
        xx, yy = np.meshgrid(x, y)
        mesh = ax[0].pcolormesh(
            xx,
            yy,
            p,
            # norm=Normalize(-1e5,1e5),
            cmap="Reds",
            alpha=0.5,
        )
        cbar = fig.colorbar(mesh, ax=ax[0])
        cbar.set_label(self.num_bursts_col)
        p = self.df_g_tau_bursts.pivot_table(
            values=i_metric, columns=constants.E_GABA, index=constants.G_GABA
        )
        x = p.columns
        y = p.index
        xx, yy = np.meshgrid(x, y)
        mesh = ax[1].pcolormesh(
            xx, yy, p, norm=Normalize(-1e6, 1e6), cmap="Greens", alpha=0.5
        )
        cbar = fig.colorbar(mesh, ax=ax[1])
        cbar.set_label(i_metric)

        p = self.df_g_tau.pivot_table(
            values=constants.I_GABA,
            columns=constants.E_GABA,
            index=constants.G_GABA,
        )
        x = p.columns
        y = p.index
        xx, yy = np.meshgrid(x, y)
        mesh = ax[2].pcolormesh(
            xx, yy, p, norm=Normalize(-1e4, 1e4), cmap="Blues", alpha=0.5
        )
        cbar = fig.colorbar(mesh, ax=ax[2])
        cbar.set_label(constants.I_GABA)
        # ax.set_xscale('log')
        ax[0].set_yscale("log")

        ax[0].set_ylabel(constants.G_GABA)
        ax[1].set_ylabel(constants.G_GABA)
        ax[2].set_ylabel(constants.G_GABA)
        ax[2].set_xlabel(constants.E_GABA)

        self.figs.append(fig)

    def plot_igaba(self, fig=None, ax=None, cax=None, min_s=None):
        from scipy import stats

        if min_s is None:
            fig_size = fig.get_size_inches()
            min_s = fig_size[0] * fig_size[1]
        sizes = (min_s / 2, 10 * min_s)

        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)
        i_metric = sum_igaba
        if i_metric == sum_igaba:
            data = self.df_g_tau_bursts[
                (self.df_g_tau_bursts[i_metric] < 1e6)
                & (self.df_g_tau_bursts[i_metric] > -1e6)
            ]
        elif i_metric == mean_igaba:
            data = self.df_g_tau_bursts[
                (self.df_g_tau_bursts[i_metric] < 100)
                & (self.df_g_tau_bursts[i_metric] > -100)
            ]
        else:
            data = self.df_g_tau_bursts
        data2 = self.df_g_E_bursts
        if len(self.seeds) > 1:
            data = (
                data[
                    [
                        constants.G_GABA,
                        constants.TAU_KCC2,
                        constants.E_GABA,
                        sum_igaba,
                        mean_igaba,
                        self.num_bursts_col,
                    ]
                ]
                .groupby(by=[constants.G_GABA, constants.TAU_KCC2], as_index=False)
                .mean()
            )
            data2 = (
                data2[
                    [
                        constants.G_GABA,
                        constants.EGABA,
                        sum_igaba,
                        mean_igaba,
                        self.num_bursts_col,
                    ]
                ]
                .groupby(by=constants.G_GABA, as_index=False)
                .mean()
            )
        combined_data = pd.concat([data, data2], axis=0)
        sns.regplot(
            x=i_metric,
            y=self.num_bursts_col,
            marker="None",
            color="k",
            data=combined_data,
            ax=ax,
        )
        r = stats.linregress(
            combined_data[i_metric], combined_data[self.num_bursts_col]
        )
        ax.annotate(
            f"$R^2$ = {r.rvalue ** 2:.2f} (p = {r.pvalue:.2g})",
            xy=(0, r.intercept),
            xytext=(-0, 15),
            fontsize="xx-small",
            ha="right",
            # arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3, rad=-0.1")
        )
        norm = LogNorm(data[constants.G_GABA].min(), data[constants.G_GABA].max())
        sns.scatterplot(
            x=i_metric,
            y=self.num_bursts_col,
            size=constants.TAU_KCC2,
            sizes=sizes,
            hue=constants.G_GABA,
            hue_norm=norm,
            palette="Greens",
            # marker='.',
            data=data,
            legend="full",
            # ec='None',
            ax=ax,
            # clip_on=False
        ).legend(frameon=False, fontsize="x-small", loc=(1, 0))
        sns.scatterplot(
            x=i_metric,
            y=self.num_bursts_col,
            hue=constants.G_GABA,
            hue_norm=norm,
            palette="Greens",
            marker="s",
            # ec='k',
            legend=False,
            data=data2,
            ax=ax,
            zorder=-99,
        )
        lines, labels = ax.get_legend_handles_labels()

        # find label that has τKCC2
        idx = labels.index(constants.TAU_KCC2) + 1
        # add white edge to each line
        for line in lines:
            line.set_edgecolor("w")
        ax.legend(
            lines[idx:],
            labels[idx:],
            loc=(1.0, 0.0),
            fontsize="x-small",
            frameon=False,
            handletextpad=0.1,
            title=constants.TAU_KCC2,
            title_fontsize="x-small",
        )

        ax.set_ylim(0, data[self.num_bursts_col].max() * 1.1)
        ax.ticklabel_format(
            axis="x", style="scientific", scilimits=(-5, 2), useMathText=True
        )
        if fig is not None:
            fig.tight_layout()
            if cax is None:
                cbar = fig.colorbar(ScalarMappable(cmap="Greens", norm=norm), ax=ax)
            else:
                cbar = fig.colorbar(
                    ScalarMappable(cmap="Greens", norm=norm),
                    cax=cax,
                    orientation="horizontal",
                )
            cbar.set_label(constants.G_GABA)
            cbar.outline.set_visible(False)
        # ax.set_xscale("symlog")

    def plot_staticegaba(self, fig=None, ax=None, norm=None, cmap=None):
        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)
        if norm is None:
            norm = settings.COLOR.EGABA_SM.norm
        if cmap is None:
            cmap = settings.COLOR.EGABA_SM.cmap
        # bursts v G with EGABA
        wide_df = self.df_g_E_bursts.pivot_table(
            index=constants.G_GABA,
            columns=constants.EGABA,
            values=self.num_bursts_col,
            fill_value=0,
        )
        wide_df.index = wide_df.index.astype(int)
        wide_df.columns = wide_df.columns.astype(int)
        sns.lineplot(
            x=constants.G_GABA,
            y=self.num_bursts_col,
            hue=constants.EGABA,
            hue_order=self.df_g_E_bursts[constants.EGABA].unique(),
            hue_norm=norm,
            palette=cmap,
            data=wide_df.reset_index().melt(
                id_vars=constants.G_GABA, value_name=self.num_bursts_col
            ),
            ax=ax,
            legend=False,
        )
        sns.lineplot(
            x=constants.G_GABA,
            y=self.num_bursts_col,
            hue=constants.EGABA,
            hue_order=self.df_g_E_bursts[constants.EGABA].unique(),
            hue_norm=norm,
            palette=cmap,
            data=self.df_g_E_bursts,
            ax=ax,
            legend=False,
        )
        if fig is not None:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax)
            cbar.set_label(f"{constants.EGABA} (mV)")
            cbar.outline.set_visible(False)
        # cbar.set_ticks(np.arange(self.EGABA_0, self.EGABA_end + self.mv_step, self.mv_step), minor=True)
        if self.gGABAsvEGABA[-1] > 100:
            ax.set_xscale("log")

    def plot_heatmap(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)
        wide_df = self.df_g_E_bursts.pivot_table(
            index=constants.G_GABA,
            columns=constants.EGABA,
            values=self.num_bursts_col,
            fill_value=0,
        )
        wide_df.index = wide_df.index.astype(int)
        wide_df.columns = wide_df.columns.astype(int)
        sns.heatmap(
            wide_df.T[::-1],
            cmap="viridis",
            ax=ax,
            cbar_kws=dict(label=self.num_bursts_col),
            vmin=0,
        )

    def jointplot(self):
        df_g_tau = self.df_g_tau
        df_g_tau_bursts = self.df_g_tau_bursts
        num_bursts_col = self.num_bursts_col

        # correlation between EGABA and TAU as a function of G plus densities for TAU and EGABA
        fig, axs_d = plt.subplots(
            2,
            2,
            sharex="col",
            sharey="row",
            gridspec_kw=dict(
                width_ratios=[1, 0.3], height_ratios=[0.3, 1], hspace=0.15, wspace=0.15
            ),
        )
        self.figs.append(fig)
        gs = axs_d[0, 1].get_gridspec()
        axs_d[0, 1].remove()
        axs_d[0, 1] = fig.add_subplot(gs[1])
        log_tau = f"log {constants.TAU_KCC2}"
        df_g_tau[log_tau] = np.log10(df_g_tau[constants.TAU_KCC2])
        g_gabas = df_g_tau[constants.G_GABA].unique()
        palette = sns.color_palette("Greens", len(g_gabas))
        sns.lineplot(
            x=log_tau,
            y=constants.EGABA,
            hue=constants.G_GABA,
            palette=palette,
            marker=".",
            data=df_g_tau,
            ax=axs_d[1, 0],
            legend="full",
        )
        handles, labels = axs_d[1, 0].get_legend_handles_labels()
        axs_d[1, 0].legend(
            handles[1:][::-1],
            labels[1:][::-1],
            ncol=1,
            loc="upper right",
            bbox_to_anchor=(1, 1),
            borderaxespad=0,
            borderpad=0,
            frameon=False,
            fontsize="x-small",
            title=constants.G_GABA,
            title_fontsize="small",
        )
        for i, g in enumerate(g_gabas):
            gdf = df_g_tau[df_g_tau[constants.G_GABA] == g]
            gbursts = df_g_tau_bursts[df_g_tau_bursts[constants.G_GABA] == g]
            # print(gbursts[[constants.TAU_KCC2,num_bursts_col]])
            total = np.sum(gbursts[num_bursts_col])
            sns.kdeplot(
                gdf[log_tau],
                bw=0.1,
                color=palette[i],
                ax=axs_d[0, 0],
                shade=True,
                legend=True,
                label=total,
                zorder=-i,
            )
            sns.kdeplot(
                gdf[constants.EGABA],
                color=palette[i],
                vertical=True,
                ax=axs_d[1, 1],
                shade=True,
                legend=False,
                zorder=-i,
            )
        axs_d[0, 0].legend(
            ncol=len(g_gabas) // 2,
            borderaxespad=0,
            borderpad=0,
            frameon=False,
            fontsize="x-small",
            title=num_bursts_col,
            title_fontsize="small",
        ).remove()

        group_g = (
            df_g_tau_bursts[[constants.G_GABA, num_bursts_col]]
            .groupby(constants.G_GABA, as_index=False)
            .sum()
        )
        sns.scatterplot(
            x=constants.G_GABA,
            y=num_bursts_col,
            hue=constants.G_GABA,
            palette=palette,
            data=group_g,
            ax=axs_d[0, 1],
            clip_on=False,
            zorder=99,
            legend=False,
        )

        tau_KCC2_list = df_g_tau[constants.TAU_KCC2].unique()
        log_tau_KCC2_list = df_g_tau[log_tau].unique()
        axs_d[1, 0].set_xticks(log_tau_KCC2_list)
        axs_d[1, 0].set_xticks([], minor=True)
        axs_d[1, 0].set_xticklabels([f"{tau:.0f}" for tau in tau_KCC2_list])
        axs_d[1, 0].set_xlabel(constants.TAU_KCC2)
        # axs_d[0,0].set_xlim(log_tau_KCC2_list[0])
        sns.despine(ax=axs_d[0, 0], left=True)
        axs_d[0, 0].set_yticks([])
        sns.despine(ax=axs_d[1, 1], bottom=True)
        axs_d[1, 1].set_xticks([])
        adjust_spines(axs_d[0, 1], ["bottom", "right"])
        sns.despine(ax=axs_d[0, 1], top=False, right=False)
        axs_d[0, 1].set_xticks(g_gabas, minor=True)
        axs_d[0, 1].set_title("Total area", fontsize="small")
        axs_d[0, 1].set_ylim(0)


if __name__ == "__main__":
    gve = Gve(
        seeds=(None, 1234, 5678),
        gGABAsvEGABA=sorted(
            set(
                np.append(
                    np.round(np.arange(0, 100.0001, 10), 0),
                    np.geomspace(10, 1000, 11).round(0),
                )
            )
        ),
        gGABAs=np.geomspace(10, 1000, 11).round(0),
        tau_KCC2s=[3.75, 7.5] + settings.TAU_KCC2_LIST,
    )
    gve.run()
    gve.process()
    gve.plot()
    gve.save_figure(file_formats=("pdf", "jpg"), figs=gve.figs, close=True)
    # zoom
    # gve = Gve(
    #         gGABAs=settings.G_GABA_LIST,
    #         tau_KCC2s=settings.TAU_KCC2_LIST[::2],
    #         )
    # gve.run()
    # gve.process()
    # gve.plot()
    # gve.sim_name = gve.fig_name + "_supp"
    # gve.save_figure(figs=gve.figs, use_args=True, close=True)
    # plt.show()
