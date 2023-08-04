import itertools
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brian2 import ms
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from scipy import interpolate, stats
from tqdm import tqdm

import settings
from core.analysis import burst_stats
from core.lrdfigure import MultiRunFigure, time_unit
from settings import logging
from style import constants
from style.axes import adjust_spines
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

        self.df_g_E = None
        self.df_g_E_bursts = None
        self.df_g_tau = None
        self.df_g_tau_bursts = None
        self.num_bursts_col = None
        self.sum_igaba = None
        self.mean_igaba = None

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
        import os

        self.sum_igaba = sum_igaba
        self.mean_igaba = mean_igaba
        bin_size = self.time_per_value
        self.num_bursts_col = num_bursts_col = f"Number of bursts\n(per {bin_size} s)"

        processing_hash = self.hash_extra(extra=f"processing - {bin_size}") + ".h5"
        folder = f"temp/ggaba_e-processing-{processing_hash}"

        if os.path.exists(f"{folder}/df_g_tau_bursts" + processing_hash):
            self.df_g_E = pd.read_hdf(
                f"{folder}/df_g_E" + processing_hash, key="df_g_E"
            )
            self.df_g_E_bursts = pd.read_hdf(
                f"{folder}/df_g_E_bursts" + processing_hash, key="df_g_E_bursts"
            )
            self.df_g_tau = pd.read_hdf(
                f"{folder}/df_g_tau" + processing_hash, key="df_g_tau"
            )
            self.df_g_tau_bursts = pd.read_hdf(
                f"{folder}/df_g_tau_bursts" + processing_hash, key="df_g_tau_bursts"
            )
            return

        logger.info("Processing bursts")
        run_idxs = list(range(len(self.seeds)))
        T = np.round(self.df.index.values[-1])

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

        gaba_run = list(itertools.product(self.gGABAsvEGABA, run_idxs))
        for gGABA, run_idx in tqdm(gaba_run, desc="Static Cl", leave=False):
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

        gaba_kcc2_run = list(itertools.product(self.gGABAs, self.tau_KCC2s, run_idxs))
        for gGABA, tau_KCC2, run_idx in tqdm(
            gaba_kcc2_run, desc="Dynamic Cl", leave=False
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

        # save to cache
        logger.info(f"saving to processing cache - {folder}")
        Path(folder).mkdir(parents=True, exist_ok=True)

        self.df_g_E.to_hdf(
            f"{folder}/df_g_E" + processing_hash, key="df_g_E", format="table"
        )
        self.df_g_E_bursts.to_hdf(
            f"{folder}/df_g_E_bursts" + processing_hash,
            key="df_g_E_bursts",
            format="table",
        )
        self.df_g_tau.to_hdf(
            f"{folder}/df_g_tau" + processing_hash, key="df_g_tau", format="table"
        )
        self.df_g_tau_bursts.to_hdf(
            f"{folder}/df_g_tau_bursts" + processing_hash,
            key="df_g_tau_bursts",
            format="table",
        )

    def plot(self, timeit=True, egabas=None, num_bursts="mean", **kwargs):
        super().plot(**kwargs)
        if self.df_g_E is None:
            self.process()
        logger.info("plotting")
        plot_time_start = time.time()

        fig, axes = plt.subplot_mosaic(
            [
                ["static_egaba", "static_egaba_cax", ".", "i_gaba_cax"],
                ["static_egaba", "static_egaba_cax", ".", "i_gaba"],
                [".", "static_egaba_cax", ".", "i_gaba"],
                ["tau_kcc2", ".", ".", "i_gaba"],
            ],
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_half),
            gridspec_kw=dict(
                width_ratios=(1.5, 0.05, 0.01, 1),
                height_ratios=(0.1, 0.5, 0.1, 1),
                wspace=0.5,
                hspace=0.5,
                left=0.1,
                right=0.9,
                bottom=0.15,
                top=0.95,
            ),
        )

        self.figs.append(fig)
        # static_ax = fig.add_subplot(gs[0, 0])
        static_ax = axes["static_egaba"]
        self.plot_staticegaba(ax=static_ax, egabas=egabas)
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
            num_bursts=num_bursts,
            **kwargs,
        )
        tau_ax.set_xlabel(f"{constants.G_GABA} (nS)")

        vline_kwargs = dict(c=settings.COLOR.K, ls="-", lw=1)
        static_ax.axvline(50, zorder=1, **vline_kwargs)
        tau_ax.axvline(50, zorder=1, **vline_kwargs)

        # igaba_ax = fig.add_subplot(gs[1, -2])
        # cax = fig.add_subplot(gs[1, -1])
        igaba_ax = axes["i_gaba"]
        igaba_cax = axes["i_gaba_cax"]
        self.plot_igaba(fig=fig, ax=igaba_ax, cax=igaba_cax, num_bursts=num_bursts)

        static_ax.set_title(
            constants.STATIC_CHLORIDE_STR_ABBR,
            fontsize="medium",
            va="top",
            ha="center",
            loc="left",
        )
        tau_ax.set_title(
            constants.DYNAMIC_CHLORIDE_STR_ABBR,
            fontsize="medium",
            va="bottom",
            ha="center",
            loc="left",
        )

        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self

    def plot_taukcc2(
        self,
        ax=None,
        cax=None,
        min_s=None,
        num_bursts="mean",
        plot_3d=False,
        mesh_norm=None,
        mesh_cmap=None,
        bursts_max: Optional[float] = None,
        **kwargs,
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
        if mesh_norm is None:
            mesh_norm = settings.COLOR.EGABA_SM.norm
        if mesh_cmap is None:
            mesh_cmap = settings.COLOR.EGABA_SM.cmap
            mesh_cmap = ListedColormap(
                sns.light_palette(
                    settings.COLOR.EGABA, self.num_ecl_steps * 2, reverse=True
                )
            )
        else:
            mesh_cmap = mesh_cmap
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
            .rolling(window=1, win_type="triang", center=True, closed="left")
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
                cmap=mesh_cmap,
                norm=mesh_norm,
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
                norm=mesh_norm,
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
                XX,
                YY,
                ZZ.T,
                cmap=mesh_cmap,
                norm=mesh_norm,
                zorder=-99,
                rasterized=True,
            )
            # plot EGABA = -60 mV line
            ax_g_v_tau.plot(s, c="k", ls="--", zorder=1)

            # get average number of bursts per g_gaba, tau_kcc2
            mean_bursts_per_g_gaba_tau_kcc2 = self.df_g_tau_bursts.groupby(
                [constants.G_GABA, constants.TAU_KCC2], as_index=False
            ).mean(numeric_only=True)
            max_bursts_per_g_gaba_tau_kcc2 = self.df_g_tau_bursts.groupby(
                [constants.G_GABA, constants.TAU_KCC2], as_index=False
            ).max(numeric_only=True)
            mean_bursts_per_g_gaba_tau_kcc2[
                self.num_bursts_col
            ] = mean_bursts_per_g_gaba_tau_kcc2[self.num_bursts_col].round(1)
            if num_bursts == "max":
                data = max_bursts_per_g_gaba_tau_kcc2
            elif num_bursts == "mean":
                data = mean_bursts_per_g_gaba_tau_kcc2
            else:
                raise ValueError(
                    f"num_bursts must be 'max' or 'mean', not {num_bursts}"
                )

            sizes = (min_s / 2, 5 * min_s)
            if bursts_max is None:
                bursts_max = max(data[self.num_bursts_col])
                bursts_max_label = bursts_max
            else:
                bursts_max_label = f">{bursts_max}"
            hue_norm = Normalize(1, bursts_max)
            size_norm = Normalize(1, bursts_max)
            sns.scatterplot(
                y=constants.TAU_KCC2,
                x=constants.G_GABA,
                hue=self.num_bursts_col,
                hue_norm=hue_norm,
                size=self.num_bursts_col,
                size_norm=size_norm,
                sizes=sizes,
                # palette=cmap,
                palette="rocket",
                #  alpha=0.8,
                # set style sothat legend has the right marker
                style=self.num_bursts_col,
                markers="o",
                data=data,
                legend="full",
                ec="w",
                ax=ax_g_v_tau,
                zorder=99,
                clip_on=False,
            )
            handles, labels = ax_g_v_tau.get_legend_handles_labels()
            # only include sizes (exclude egaba)
            if self.num_bursts_col in labels:
                idx_bursts = labels.index(self.num_bursts_col) + 1
            else:
                idx_bursts = 0
            handles, labels = handles[idx_bursts:], labels[idx_bursts:]

            # choose first, last and some middle labels
            labels = [
                labels[0],
                labels[round(len(labels) * 1 / 4)],
                labels[round(len(labels) * 1 / 2)],
                labels[round(len(labels) * 3 / 4)],
                # labels[round(len(labels) * 7 / 8)],
                bursts_max_label,
            ]
            handles = [
                handles[0],
                handles[round(len(handles) * 1 / 4)],
                handles[round(len(handles) * 1 / 2)],
                handles[round(len(handles) * 3 / 4)],
                # handles[round(len(labels) * 7 / 8)],
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
            cbar = fig.colorbar(mesh, ax=ax_g_v_tau, cmap=mesh_cmap, norm=mesh_norm)
        else:
            cbar = fig.colorbar(
                mesh,
                cax=cax,
                cmap=mesh_cmap,
                norm=mesh_norm,
            )
        cbar.set_label(f"{constants.EGABA} (mV)")
        cbar.outline.set_visible(False)
        # cbar.set_ticks(np.arange(norm.vmin, norm.vmax+1, 10))
        cbar.minorticks_on()

    def i_explore(
        self, i_metric: str = None, index=constants.E_GABA, columns=constants.G_GABA
    ):
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
            columns=columns,
            index=index,
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
            values=i_metric,
            columns=columns,
            index=index,
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
            columns=columns,
            index=index,
        )
        x = p.columns
        y = p.index
        xx, yy = np.meshgrid(x, y)
        mesh = ax[2].pcolormesh(
            xx, yy, p, norm=Normalize(-1e4, 1e4), cmap="Blues", alpha=0.5
        )
        cbar = fig.colorbar(mesh, ax=ax[2])
        cbar.set_label(constants.I_GABA)

        if index == constants.G_GABA:
            ax[0].set_yscale("log")
        if columns == constants.G_GABA:
            ax[0].set_xscale("log")
            for a in ax:
                a.set_xlabel(constants.G_GABA)

        for a in ax:
            a.set_ylabel(index)
        ax[0].set_xlabel(columns)

        self.figs.append(fig)

    def plot_igaba(
        self,
        fig=None,
        ax=None,
        cax=None,
        min_s=None,
        num_bursts="mean",
        i_metric=sum_igaba,
    ):
        if min_s is None:
            fig_size = fig.get_size_inches()
            min_s = fig_size[0] * fig_size[1]
        sizes = (min_s / 2, 10 * min_s)

        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)

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
        dynamic_value = f"Dynamic {constants.CL}"
        static_value = f"Static {constants.CL}"
        data[constants.ECL] = dynamic_value
        data2[constants.ECL] = static_value

        data[constants.TAU_KCC2] = data[constants.TAU_KCC2].astype(int)
        # assign middle tau to static for sizing in graph
        all_taus = data[constants.TAU_KCC2].unique()
        middle_tau = all_taus[len(all_taus) // 2]
        data2[constants.TAU_KCC2] = middle_tau

        combined_data = pd.concat([data, data2], axis=0)
        sns.regplot(
            x=i_metric,
            y=self.num_bursts_col,
            marker="None",
            color="k",
            data=data,
            ax=ax,
        )
        r = stats.linregress(data[i_metric], data[self.num_bursts_col])
        ax.annotate(
            f"$R^2$ = {r.rvalue ** 2:.2f} (p = {r.pvalue:.2g})",
            xy=(1, 1),
            xytext=(-0, 15),
            fontsize="xx-small",
            ha="right",
            va="top",
            # arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3, rad=-0.1")
        )

        norm = settings.G_GABA_Norm(
            50, data[constants.G_GABA].min(), data[constants.G_GABA].max()
        )

        sns.scatterplot(
            x=i_metric,
            y=self.num_bursts_col,
            size=constants.TAU_KCC2,
            sizes=sizes,
            hue=constants.G_GABA,
            hue_norm=norm,
            palette=settings.COLOR.G_GABA_SM.get_cmap(),
            style=constants.ECL,
            style_order=[dynamic_value, static_value],
            # marker='.',
            markers={static_value: "s", dynamic_value: "D"},
            data=combined_data,
            legend="full",
            # ec='None',
            ax=ax,
            # clip_on=False
        ).legend(frameon=False, fontsize="x-small", loc=(1, 0))
        # plot static data
        # sns.scatterplot(
        #     x=i_metric,
        #     y=self.num_bursts_col,
        #     hue=constants.G_GABA,
        #     hue_norm=norm,
        #     palette=settings.COLOR.G_GABA_SM.get_cmap(),
        #     marker="D",
        #     # ec='k',
        #     legend=False,
        #     data=data2,
        #     ax=ax,
        #     zorder=-99,
        # )
        legend_path_collections, labels = ax.get_legend_handles_labels()
        # find label that has τKCC2
        idx = labels.index(constants.TAU_KCC2) + 1
        # find label that has ECl
        idx_ecl = labels.index(constants.ECL)
        # first ecl label is dynamic (based on style_order)
        diamond_path = legend_path_collections[idx_ecl + 1].get_paths()
        # add white edge to each line
        for i, pc in enumerate(legend_path_collections):
            pc.set_edgecolor("w")
            if i >= idx_ecl:
                continue
            pc.set_paths(diamond_path)
        ax.legend(
            legend_path_collections[idx:idx_ecl] + legend_path_collections[-1:],
            labels[idx:idx_ecl] + labels[-1:],
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
                cbar = fig.colorbar(
                    ScalarMappable(cmap=settings.COLOR.G_GABA_SM.get_cmap(), norm=norm),
                    ax=ax,
                )
            else:
                cbar = fig.colorbar(
                    ScalarMappable(cmap=settings.COLOR.G_GABA_SM.get_cmap(), norm=norm),
                    cax=cax,
                    orientation="horizontal",
                )
            cbar.set_label(constants.G_GABA)
            cbar.outline.set_visible(False)
        # ax.set_xscale("symlog")

    def plot_staticegaba(self, fig=None, ax=None, norm=None, cmap=None, egabas=None):
        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)
        if norm is None:
            norm = settings.COLOR.EGABA_SM.norm
        if cmap is None:
            cmap = settings.COLOR.EGABA_SM.cmap
        if egabas is None:
            egabas = self.df_g_E_bursts[constants.EGABA].unique()
        elif isinstance(egabas, int):
            _egaba_values = np.arange(self.EGABA_0, self.EGABA_end, self.mv_step)
            egabas = _egaba_values[:: len(_egaba_values) // egabas]

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
            hue_order=egabas,
            hue_norm=norm,
            palette=cmap,
            data=wide_df.reset_index()
            .melt(id_vars=constants.G_GABA, value_name=self.num_bursts_col)
            .query(f"{constants.EGABA} in {egabas.tolist()}"),
            ax=ax,
            legend=False,
        )
        # plot again for error bars (previous plot omits error bars but includes 0 values)
        sns.lineplot(
            x=constants.G_GABA,
            y=self.num_bursts_col,
            hue=constants.EGABA,
            hue_order=egabas,
            hue_norm=norm,
            palette=cmap,
            data=self.df_g_E_bursts.query(f"{constants.EGABA} in {egabas.tolist()}"),
            ax=ax,
            legend=True,
            err_style="bars",
            errorbar="se",
        )
        if fig is not None:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label(f"{constants.EGABA} (mV)")
            cbar.outline.set_visible(False)
        # cbar.set_ticks(np.arange(self.EGABA_0, self.EGABA_end + self.mv_step, self.mv_step), minor=True)
        if self.gGABAsvEGABA[-1] > 100:
            ax.set_xscale("log")

    def plot_heatmap(
        self,
        static=False,
        x=constants.G_GABA,
        y=constants.EGABA,
        c="Number of bursts\n(per 60 s)",
        agg="mean",
        cmap="viridis",
        ax=None,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)
        if static:
            df = self.df_g_E_bursts
        else:
            df = self.df_g_tau_bursts

        try:
            if agg == "mean":
                df = df.groupby([x, y], as_index=False).mean(numeric_only=True)
            else:
                df = df.groupby([x, y], as_index=False).agg(agg, numeric_only=True)

            if "query" in kwargs:
                df = df.query(kwargs.pop("query"))

            wide_df = df.pivot_table(
                index=x,
                columns=y,
                values=c,
                fill_value=np.nan,
            )
        except KeyError as err:
            logger.error(err)
            logger.error(f"possible values are {df.columns}")
            raise err

        wide_df.index = wide_df.index.astype(int)
        wide_df.columns = wide_df.columns.astype(int)
        mask = kwargs.pop("mask", None)
        if mask is not None:
            mask = eval(f"wide_df.T[::-1] {mask} ")
        sns.heatmap(
            wide_df.T[::-1],
            cmap=cmap,
            ax=ax,
            cbar_kws=dict(label=c),
            norm=LogNorm()
            if c == constants.TAU_KCC2 or c == constants.G_GABA
            else None,
            # vmin=0,
            mask=mask,
            **kwargs,
        )
        return wide_df.T[::-1]

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
        palette = sns.color_palette(settings.COLOR.G_GABA_SM.get_cmap(), len(g_gabas))
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

        sns.histplot(
            data=df_g_tau,
            x=log_tau,
            hue=constants.G_GABA,
            palette=palette,
            ax=axs_d[0, 0],
            multiple="stack",
            stat="density",
            fill=True,
            legend=False,
        )
        sns.histplot(
            data=df_g_tau,
            y=constants.EGABA,
            hue=constants.G_GABA,
            palette=palette,
            ax=axs_d[1, 1],
            multiple="stack",
            stat="density",
            fill=True,
            legend=False,
        )
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
    # extend tau_KCC2 list to lower values, at the same ratio as existing values
    tau_KCC2_list = settings.TAU_KCC2_LIST

    ratio = tau_KCC2_list[1] / tau_KCC2_list[0]
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list

    gve = Gve(
        seeds=(None, 1234, 5678, 1426987, 86751, 1010, 876, 12576, 9681, 814265),
        gGABAsvEGABA=sorted(
            set(
                np.append(
                    np.round(np.arange(0, 100.0001, 10), 0),
                    np.geomspace(10, 1000, 11).round(0),
                )
            )
        ),
        gGABAs=np.geomspace(10, 1000, 11).round(0),
        tau_KCC2s=tau_KCC2_list,
    )
    gve.run()
    gve.process()
    gve.plot(egabas=5)
    gve.save_figure(figs=gve.figs, close=True)
