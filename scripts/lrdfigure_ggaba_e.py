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
from style import text
from style.axes import adjust_spines
from style.color import COLOR, G_GABA_Norm
from style.text_math import math_fix

logger = logging.getLogger(__name__)
round_EGABA = f"{text.E_GABA}"
sum_igaba = math_fix(
    r"$\sum_b^{bursts} \int_{b_{start} - 1 s}^{b_{start}} " f"{text.I_GABA} (pA)$"
)
mean_igaba = math_fix(f"$\langle {text.I_GABA} \\rangle (pA)$")  # noqa: W605


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
            OrderedDict(g_GABA_max={"range": self.gGABAs, "title": text.G_GABA}),
            **kwargs,
        )

        self.df_g_E = None
        self.df_g_E_bursts = None
        self.df_g_tau = None
        self.df_g_tau_bursts = None
        self.num_bursts_col = None

    def run(
        self,
        subsample=10,
        time_per_value=60,
        EGABA_0=-74,
        EGABA_end=-40,
        mv_step=2,
        **kwargs,
    ):
        logger.info(f"PART 1 - static Chloride\n{'*'*20}")
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
                g_GABA_max={"range": self.gGABAsvEGABA, "title": text.G_GABA},
            ),
            seeds=self.seeds,
            default_params=dict(
                manual_cl=manual_cl, duration=duration, num_ecl_steps=num_ecl_steps
            ),
            **kwargs,
        )
        super().run(subsample=subsample, **kwargs)

        self.df_main, self.df = self.df, None

        logger.info(f"PART 2 - dynamic Chloride\n{'*'*20}")
        # set reasonable starting ECl
        EGABA_0 = -74
        EGABA_end = -40
        ecl_0 = round((EGABA_0 - phco3 * ehco3) / pcl, 2)
        ecl_end = round((EGABA_end - phco3 * ehco3) / pcl, 2)
        kwargs["E_Cl_0"] = ecl_0 / 2 + ecl_end / 2  # start at mid-point between the 2
        kwargs["E_Cl_end"] = ecl_end  # TODO: remove (used for cached results)

        super().__init__(
            OrderedDict(
                g_GABA_max={"range": self.gGABAs, "title": text.G_GABA},
                tau_KCC2_E={"range": self.tau_KCC2s, "title": text.TAU_KCC2},
                tau_KCC2_I=text.TAU_KCC2,
            ),
            seeds=self.seeds,
            default_params=dict(
                dyn_cl=True,
                duration=600,
            ),
            **kwargs,
        )

        super().run(subsample=subsample, use_vaex=True, **kwargs)
        self.dynamic, self.df = self.df, self.df_main

        return self

    def process(self):
        """
        Process the data and store the results in the cache folder. No parameters and no return type specified.

        Makes available to the instance:
        - self.df_g_E - DataFrame of g_GABA, E_GABA, total I_GABA, and burst start time for each seed and each burst
        - self.df_g_E_bursts - DataFrame of g_GABA, E_GABA, and number of bursts for each seed
        - self.df_g_tau - DataFrame of g_GABA, tau_KCC2, E_GABA (recorded), total I_GABA before each burst,
            and burst start time for each seed and each burst
        - self.df_g_tau_bursts - DataFrame of g_GABA, tau_KCC2, E_GABA (recorded), I_GABA (mean and sum),
            and number of bursts for each seed
        """
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
        mid_bin_idx = int(len(bins) / 2)
        _t_offset = 1 * int(time_unit / ms)

        ###############
        # static EGABA
        ###############
        df_g_E = pd.DataFrame(
            columns=[
                "run_idx",
                text.I_GABA,
                text.EGABA,
                text.G_GABA,
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
                    (self.df["gGABA"] == gGABA) & (self.df["run_idx"] == run_idx)
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
                text.I_GABA,
                text.EGABA,
                text.G_GABA,
                text.TAU_KCC2,
                "Burst start time (s)",
            ]
        )

        gaba_kcc2_run = list(itertools.product(self.gGABAs, self.tau_KCC2s, run_idxs))
        for gGABA, tau_KCC2, run_idx in tqdm(
            gaba_kcc2_run, desc="Dynamic Cl", leave=False
        ):
            if isinstance(self.dynamic, pd.DataFrame):
                instance_df = self.dynamic[gGABA, tau_KCC2, run_idx].sort_index()
            else:
                instance_df = (
                    self.dynamic[
                        (self.dynamic[text.G_GABA] == gGABA)
                        & (self.dynamic[text.TAU_KCC2] == tau_KCC2)
                        & (self.dynamic["run_idx"] == run_idx)
                    ]
                    .to_pandas_df(
                        ["E_GABA_all", "I_GABA_all", "r_all"], index_name="Time"
                    )
                    .sort_index()
                )

            df_egaba = instance_df["E_GABA_all"]
            df_igaba = instance_df["I_GABA_all"]
            df_rates = instance_df["r_all"]
            burst_start_ts, burst_end_ts = burst_stats(
                df_rates,
                rate_std_thresh=2,
                time_unit=time_unit,
                plot_fig=False,
            )
            # only consider once network has reached steady-state
            burst_start_ts = burst_start_ts[burst_start_ts >= bins[mid_bin_idx]]
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
            df_g_E.groupby([text.EGABA, text.G_GABA, "bin", "run_idx"])
            .agg(["count", "sum", "mean"])
            .dropna()
            .drop(
                [
                    (text.I_GABA, "count"),
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
            df_g_E_bursts[text.EGABA], bins=[-100, -60, 0], labels=[True, False]
        )

        df_g_tau[round_EGABA] = df_g_tau[text.EGABA].round(0)

        df_g_tau_bursts = (
            df_g_tau.groupby(
                [round_EGABA, text.G_GABA, text.TAU_KCC2, "bin", "run_idx"]
            )
            .agg(["count", "sum", "mean"])
            .dropna()
            .drop(
                [
                    (text.I_GABA, "count"),
                    text.EGABA,
                    ("Burst start time (s)", "sum"),
                    ("Burst start time (s)", "mean"),
                ],
                axis=1,
            )
        )
        # change from hierarchical index to flat index and change name
        df_g_tau_bursts.columns = [sum_igaba, mean_igaba, num_bursts_col]
        df_g_tau_bursts.reset_index(inplace=True)
        # df_mean = df_g_tau.groupby([round_EGABA, text.G_GABA, text.TAU_KCC2, 'bin', 'run_idx'],
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

    def plot(
        self, timeit=True, egabas=None, num_bursts="mean", i_metric="diagram", **kwargs
    ):
        """
        Plot the figure.

        :param timeit: Whether to measure the time it takes to plot the data. Default is True.
        :param egabas: A list of GABA conductance values. Default is None (all GABA conductance values in init).
        :param num_bursts: How to consider the number of bursts. Can be "mean" or "sum. Default is "mean".
        :param i_metric: The metric to use for the GABA current calculation. Can be "mean" or "sum" or "diagram".
            Default is mean_igaba constant (defined at top of file). This is deprecated and "diagram" is used instead
            to leave open space for a diagram.
        :param kwargs: Additional keyword arguments to pass to the super().plot() and sub plotting method.
        :return: The updated instance of the class.

        :note: If you want to plot the data, you must call process() first.
        :note: i_metric is deprecated as it doesn't make sense to regress over GABA current for the number of bursts
            because GABA current is dependent on the number of bursts
            (even for mean_igaba as it is defined in process()).

        """
        super().plot(**kwargs)
        if self.df_g_E is None:
            self.process()
        if i_metric == "sum":
            i_metric = sum_igaba
        elif i_metric == "mean":
            i_metric = mean_igaba
        assert i_metric in (
            mean_igaba,
            sum_igaba,
            "diagram",
        ), "i_metric must be 'mean' or 'sum' or 'diagram'"

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
        self.plot_staticegaba(
            ax=static_ax,
            egabas=egabas,
            cmap=kwargs.get("mesh_cmap"),
            norm=kwargs.get("mesh_norm"),
        )
        static_ax.set_xlim(10, xmax=1000)
        static_ax.set_ylim(0, ymax=13)
        static_ax.set_xlabel(f"{text.G_GABA} (nS)")
        # sns.scatterplot(x=text.G_GABA, y=round_EGABA,
        #                 hue=text.TAU_KCC2,
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
            # min_s=fig_size[0] * fig_size[1], # moved to plot_taukcc2
            num_bursts=num_bursts,
            **kwargs,
        )
        tau_ax.set_xlabel(f"{text.G_GABA} (nS)")
        tau_ax.set_ylabel(f"{text.TAU_KCC2} (s)")

        vline_kwargs = dict(c=settings.COLOR.K, ls="-", lw=1)
        static_ax.axvline(50, zorder=1, **vline_kwargs)
        tau_ax.axvline(50, zorder=1, **vline_kwargs)

        igaba_ax = axes["i_gaba"]
        igaba_cax = axes["i_gaba_cax"]
        if i_metric in {sum_igaba, mean_igaba}:
            self.plot_igaba(
                fig=fig,
                ax=igaba_ax,
                cax=igaba_cax,
                i_metric=i_metric,
            )
        else:
            # leave open space for a diagram but include colorbar
            cbar = fig.colorbar(
                ScalarMappable(
                    cmap=settings.COLOR.G_GABA_SM.get_cmap(),
                    norm=settings.COLOR.G_GABA_SM.norm,
                ),
                cax=igaba_cax,
                orientation="horizontal",
            )
            cbar.set_label(text.G_GABA)
            cbar.outline.set_visible(False)

        static_ax.set_title(
            text.STATIC_CHLORIDE_STR_ABBR,
            fontsize="medium",
            va="top",
            ha="center",
            loc="left",
        )
        tau_ax.set_title(
            text.DYNAMIC_CHLORIDE_STR_ABBR,
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
                    settings.COLOR.EGABA,
                    self.num_ecl_steps * 2,
                    reverse=settings.COLOR.EGABA_reverse,
                )
            )
        else:
            mesh_cmap = mesh_cmap
        p = self.df_g_tau.pivot_table(
            values=text.EGABA, index=text.TAU_KCC2, columns=text.G_GABA
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
        # -60 mV line
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
                index=text.TAU_KCC2,
                columns=text.G_GABA,
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
                ZZ.T,
                rcount=200,
                ccount=200,
                cmap=mesh_cmap,
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
            ax.set_xlabel("log10(g_gaba)")
            ax.set_ylabel("log10(tau_kcc2)")
            ax.set_zlabel("EGABA")
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
                [text.G_GABA, text.TAU_KCC2], as_index=False
            ).mean(numeric_only=True)
            max_bursts_per_g_gaba_tau_kcc2 = self.df_g_tau_bursts.groupby(
                [text.G_GABA, text.TAU_KCC2], as_index=False
            ).max(numeric_only=True)
            mean_bursts_per_g_gaba_tau_kcc2[self.num_bursts_col] = (
                mean_bursts_per_g_gaba_tau_kcc2[self.num_bursts_col].round(1)
            )
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
                y=text.TAU_KCC2,
                x=text.G_GABA,
                hue=self.num_bursts_col,
                hue_norm=hue_norm,
                size=self.num_bursts_col,
                size_norm=size_norm,
                sizes=sizes,
                palette=kwargs.get("cmap", settings.COLOR.NUM_BURSTS_CMAP),
                #  alpha=0.8,
                # set style so that legend has the right marker
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
            tau_KCC2_list = self.df_g_tau[text.TAU_KCC2].unique()
            ax_g_v_tau.set_yticks(tau_KCC2_list)
            ax_g_v_tau.set_yticks([], minor=True)
            ax_g_v_tau.set_yticklabels(
                [f"{t:.0f}" if t > 10 else f"{t:.1f}" for t in tau_KCC2_list]
            )
            # ax_g_v_tau.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

            ax_g_v_tau.set_ylim(tau_KCC2_list[0])
            g_gaba_list = self.df_g_tau[text.G_GABA].unique()
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
        cbar.set_label(f"{text.EGABA} (mV)")
        cbar.outline.set_visible(False)
        # cbar.set_ticks(np.arange(norm.vmin, norm.vmax+1, 10))
        cbar.minorticks_on()

    def i_explore(self, i_metric: str = None, index=text.E_GABA, columns=text.G_GABA):
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
            values=text.I_GABA,
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
        cbar.set_label(text.I_GABA)

        if index == text.G_GABA:
            ax[0].set_yscale("log")
        if columns == text.G_GABA:
            ax[0].set_xscale("log")
            for a in ax:
                a.set_xlabel(text.G_GABA)

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
        i_metric=sum_igaba,
    ):
        if min_s is None:
            fig_size = fig.get_size_inches()
            min_s = fig_size[0] * fig_size[1]
        sizes = (min_s / 2, 10 * min_s)

        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)

        data = self.df_g_tau_bursts
        # if i_metric == mean_igaba:
        #     data = data[(data[text.G_GABA] >= 40)]

        data2 = self.df_g_E_bursts
        # calculate outliers
        data["z score"] = stats.zscore(data[i_metric], nan_policy="omit")
        data = data[data["z score"].abs() < 2]
        data2["z score"] = stats.zscore(data2[i_metric], nan_policy="omit")
        data2 = data2[data2["z score"].abs() < 2]

        # calculate mean if there are multiple seeds
        if len(self.seeds) > 10:
            data = (
                data[
                    [
                        text.G_GABA,
                        text.TAU_KCC2,
                        text.E_GABA,
                        sum_igaba,
                        mean_igaba,
                        self.num_bursts_col,
                    ]
                ]
                .groupby(by=[text.G_GABA, text.TAU_KCC2], as_index=False)
                .mean()
            )
            data2 = (
                data2[
                    [
                        text.G_GABA,
                        text.EGABA,
                        sum_igaba,
                        mean_igaba,
                        self.num_bursts_col,
                    ]
                ]
                .groupby(by=text.G_GABA, as_index=False)
                .mean()
            )
        dynamic_value = f"Dynamic {text.CL}"
        static_value = f"Static {text.CL}"
        data[text.ECL] = dynamic_value
        data2[text.ECL] = static_value
        data[text.TAU_KCC2] = data[text.TAU_KCC2].astype(int)
        # assign middle tau to static for sizing in graph
        all_taus = data[text.TAU_KCC2].unique()
        middle_tau = all_taus[len(all_taus) // 2]
        data2[text.TAU_KCC2] = middle_tau

        combined_data = pd.concat([data, data2], axis=0)
        sns.regplot(
            x=i_metric,
            y=self.num_bursts_col,
            marker="None",
            color="k",
            data=data,
            ax=ax,
        )
        r = stats.linregress(data[i_metric], data[self.num_bursts_col])  # type: ignore
        ax.annotate(
            f"$R^2$ = {r.rvalue ** 2:.2f} (p = {r.pvalue:.2g})",  # type: ignore
            xy=(1, 1),
            xytext=(-0, 15),
            fontsize="xx-small",
            ha="right",
            va="top",
            # arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3, rad=-0.1")
        )

        norm = G_GABA_Norm(50, data[text.G_GABA].min(), data[text.G_GABA].max())

        sns.scatterplot(
            x=i_metric,
            y=self.num_bursts_col,
            size=text.TAU_KCC2,
            sizes=sizes,
            hue=text.G_GABA,
            hue_norm=norm,
            palette=COLOR.G_GABA_SM.get_cmap(),
            style=text.ECL,
            style_order=[dynamic_value, static_value],
            # marker='.',
            markers={static_value: "s", dynamic_value: "D"},
            data=combined_data,
            legend="full",
            # ec='None',
            alpha=0.2,
            ax=ax,
            # clip_on=False
        ).legend(frameon=False, fontsize="x-small", loc=(1, 0))
        # plot static data
        # sns.scatterplot(
        #     x=i_metric,
        #     y=self.num_bursts_col,
        #     hue=text.G_GABA,
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
        idx = labels.index(text.TAU_KCC2) + 1
        # find label that has ECl
        idx_ecl = labels.index(text.ECL)
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
            title=text.TAU_KCC2,
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
            cbar.set_label(text.G_GABA)
            cbar.outline.set_visible(False)
        # ax.set_xscale("symlog")

    def plot_staticegaba(
        self, fig=None, ax=None, norm=None, cmap=None, egabas=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            self.figs.append(fig)
        if norm is None:
            norm = settings.COLOR.EGABA_SM.norm
        if cmap is None:
            cmap = settings.COLOR.EGABA_SM.cmap
        if egabas is None:
            egabas = sorted(self.df_g_E_bursts[text.EGABA].unique())
        elif isinstance(egabas, int):
            _egaba_values = np.arange(
                self.EGABA_0, self.EGABA_end, self.mv_step
            ).tolist()
            egabas = _egaba_values[:: len(_egaba_values) // egabas]

        # bursts v G with EGABA
        wide_df = self.df_g_E_bursts.pivot_table(
            index=text.G_GABA,
            columns=text.EGABA,
            values=self.num_bursts_col,
            fill_value=0,
        )
        wide_df.index = wide_df.index.astype(int)
        wide_df.columns = wide_df.columns.astype(int)
        sns.lineplot(
            x=text.G_GABA,
            y=self.num_bursts_col,
            hue=text.EGABA,
            hue_order=egabas,
            hue_norm=norm,
            palette=cmap,
            data=wide_df.reset_index()
            .melt(id_vars=text.G_GABA, value_name=self.num_bursts_col)
            .query(f"{text.EGABA} in {egabas}"),
            ax=ax,
            legend=False,
        )
        # plot again for error bars (previous plot omits error bars but includes 0 values)
        sns.lineplot(
            x=text.G_GABA,
            y=self.num_bursts_col,
            hue=text.EGABA,
            hue_order=egabas,
            hue_norm=norm,
            palette=cmap,
            data=self.df_g_E_bursts.query(f"{text.EGABA} in {egabas}"),
            ax=ax,
            legend=(
                "brief"
                if len(egabas) == len(self.df_g_E_bursts[text.EGABA].unique())
                else "full"
            ),
            err_style="bars",
            errorbar="se",
        )
        # plot just -60mv
        sns.lineplot(
            x=text.G_GABA,
            y=self.num_bursts_col,
            c="k",
            ls="--",
            lw=2,
            data=wide_df.reset_index()
            .melt(id_vars=text.G_GABA, value_name=self.num_bursts_col)
            .query(f"{text.EGABA} in [-60]"),
            ax=ax,
            legend=False,
        )

        if fig is not None:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label(f"{text.EGABA} (mV)")
            cbar.outline.set_visible(False)
        # cbar.set_ticks(np.arange(self.EGABA_0, self.EGABA_end + self.mv_step, self.mv_step), minor=True)
        if self.gGABAsvEGABA[-1] > 100:
            ax.set_xscale("log")

    def plot_heatmap(
        self,
        static=False,
        x=text.G_GABA,
        y=text.EGABA,
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
            norm=LogNorm() if c == text.TAU_KCC2 or c == text.G_GABA else None,
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
        log_tau = f"log {text.TAU_KCC2}"
        df_g_tau[log_tau] = np.log10(df_g_tau[text.TAU_KCC2])
        g_gabas = df_g_tau[text.G_GABA].unique()
        palette = {
            g_gaba: settings.COLOR.G_GABA_SM.to_rgba(g_gaba) for g_gaba in g_gabas
        }
        sns.lineplot(
            x=log_tau,
            y=text.EGABA,
            hue=text.G_GABA,
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
            title=text.G_GABA,
            title_fontsize="small",
        )

        sns.histplot(
            data=df_g_tau,
            x=log_tau,
            hue=text.G_GABA,
            palette=palette,
            ax=axs_d[0, 0],
            multiple="stack",
            stat="density",
            fill=True,
            legend=False,
        )
        sns.histplot(
            data=df_g_tau,
            y=text.EGABA,
            hue=text.G_GABA,
            palette=palette,
            ax=axs_d[1, 1],
            multiple="stack",
            stat="density",
            fill=True,
            legend=False,
        )
        group_g = (
            df_g_tau_bursts[[text.G_GABA, num_bursts_col]]
            .groupby(text.G_GABA, as_index=False)
            .mean()
        )
        sns.scatterplot(
            x=text.G_GABA,
            y=num_bursts_col,
            hue=text.G_GABA,
            palette=palette,
            data=group_g,
            ax=axs_d[0, 1],
            clip_on=False,
            zorder=99,
            legend=False,
        )

        tau_KCC2_list = df_g_tau[text.TAU_KCC2].unique()
        log_tau_KCC2_list = df_g_tau[log_tau].unique()
        axs_d[1, 0].set_xticks(log_tau_KCC2_list)
        axs_d[1, 0].set_xticks([], minor=True)
        axs_d[1, 0].set_xticklabels([f"{tau:.0f}" for tau in tau_KCC2_list])
        axs_d[1, 0].set_xlabel(text.TAU_KCC2)
        # axs_d[0,0].set_xlim(log_tau_KCC2_list[0])
        sns.despine(ax=axs_d[0, 0], left=True)
        axs_d[0, 0].set_yticks([])
        sns.despine(ax=axs_d[1, 1], bottom=True)
        axs_d[1, 1].set_xticks([])
        adjust_spines(axs_d[0, 1], ["bottom", "right"])
        sns.despine(ax=axs_d[0, 1], top=False, right=False)
        axs_d[0, 1].set_xticks(g_gabas, minor=True)
        axs_d[0, 1].set_title("Mean area", fontsize="small")
        axs_d[0, 1].set_ylim(0)

    def supp_figure(
        self,
        ggaba_ignore=None,
        egaba_vs_ggaba_static_cl=True,
        egaba_vs_ggaba=True,
        egaba_vs_tau_kcc2=True,
        bursts_vs_egaba=True,
        bursts_vs_ggaba=True,
        bursts_vs_tau_kcc2=True,
    ):

        ggaba_ignore = (
            ggaba_ignore or []
        )  # sorted(set(np.geomspace(60, 158, 7).round(0)) - {97})
        if "$E_{GABA}$" in self.df_g_tau_bursts:
            self.df_g_tau_bursts["EGABA"] = self.df_g_tau_bursts["$E_{GABA}$"]
        df_sub_bursts = self.df_g_tau_bursts[
            ~self.df_g_tau_bursts[text.G_GABA].isin(ggaba_ignore)
        ]
        df_sub_tau = self.df_g_tau[~self.df_g_tau[text.G_GABA].isin(ggaba_ignore)]

        # fill in 0s for missing values
        for gaba, tau in itertools.product(
            df_sub_tau[text.G_GABA].unique(), df_sub_tau[text.TAU_KCC2].unique()
        ):
            entry_exists = (
                df_sub_bursts[
                    (df_sub_bursts[text.G_GABA] == gaba)
                    & (df_sub_bursts[text.TAU_KCC2] == tau)
                ].shape[0]
                > 0
            )
            if not entry_exists:
                df_sub_bursts = pd.concat(
                    [
                        df_sub_bursts,
                        pd.DataFrame(
                            {
                                text.G_GABA: gaba,
                                text.TAU_KCC2: tau,
                                self.num_bursts_col: 0,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

        fig, axs = plt.subplots(
            2,
            max(
                egaba_vs_ggaba_static_cl + egaba_vs_ggaba + egaba_vs_tau_kcc2,
                bursts_vs_egaba + bursts_vs_ggaba + bursts_vs_tau_kcc2,
            ),
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_half),
            squeeze=False,  # always use [row, col] notation
            # sharey="row",
            # gridspec_kw={"width_ratios": [1, 1, 1]},
        )

        if egaba_vs_ggaba_static_cl:
            ax = axs[0, 0]

            g_gabas = sorted(self.df_g_E_bursts[text.G_GABA].unique())
            egabas = sorted(self.df_g_E_bursts["EGABA"].unique())
            GG, EE = np.meshgrid(g_gabas, egabas)
            ZZ = EE

            ax.pcolor(
                GG, EE, ZZ, cmap=COLOR.EGABA_SM.get_cmap(), norm=COLOR.EGABA_SM.norm
            )
            sns.scatterplot(
                data=self.df_g_E_bursts[
                    self.df_g_E_bursts[text.G_GABA].isin(
                        set(self.df_g_tau_bursts[text.G_GABA].unique())
                    )
                ],
                x=text.G_GABA,
                y="EGABA",
                size=self.num_bursts_col,
                hue=self.num_bursts_col,
                size_norm=(1, 15),
                hue_norm=(1, 15),
                ax=ax,
                palette="viridis",
                clip_on=False,
                zorder=10,
                # cmap=COLOR.EGABA_SM.get_cmap()
            )

            ax.set_xscale("log")
            ax.set_xlim(10, 1000)

        if egaba_vs_ggaba:
            ax_ggaba = axs[0, int(egaba_vs_ggaba_static_cl)]

            sns.lineplot(
                data=self.df_g_tau,
                y="EGABA",
                x=text.G_GABA,
                hue=text.TAU_KCC2,
                palette=COLOR.TAU_SM.get_cmap(),
                hue_norm=COLOR.TAU_SM.norm,
                marker=".",
                ax=ax_ggaba,
                legend="full",
            )
            ax_ggaba.set_xscale("log")

        if egaba_vs_tau_kcc2:
            ax_tau = axs[0, int(egaba_vs_ggaba_static_cl + egaba_vs_ggaba)]

            sns.lineplot(
                data=self.df_g_tau,
                y="EGABA",
                x=text.TAU_KCC2,
                hue=text.G_GABA,
                palette=COLOR.G_GABA_SM.cmap,
                hue_norm=COLOR.G_GABA_SM.norm,
                marker=".",
                ax=ax_tau,
                legend="full",
            )
            taus = sorted(self.df_g_tau[text.TAU_KCC2].unique())
            ax_tau.set_xscale("log")
            tau_labels = [
                f"{tau:.0f}" if int(tau) == float(tau) else f"{tau}" for tau in taus
            ]
            ax_tau.set_xticks(taus, labels=[], minor=True)
            ax_tau.set_xticks(taus[::2], labels=tau_labels[::2])

        # self.plot_taukcc2(ax=axs[1,0], cax=axs[0, 1])

        if bursts_vs_egaba:
            ax_bursts_vs_egaba = axs[1, 0]
            sns.scatterplot(
                data=df_sub_bursts,
                y=self.num_bursts_col,
                x="EGABA",
                ax=ax_bursts_vs_egaba,
                marker=".",
                hue=text.G_GABA,
                palette=COLOR.G_GABA_SM.cmap,
                hue_norm=COLOR.G_GABA_SM.norm,
                size=text.TAU_KCC2,
            )

            reg_data = df_sub_bursts[["EGABA", self.num_bursts_col]].dropna()
            sns.regplot(
                data=reg_data,
                y=self.num_bursts_col,
                x="EGABA",
                ax=ax_bursts_vs_egaba,
                scatter=False,
                color="k",
            )
            r = stats.linregress(reg_data["EGABA"], reg_data[self.num_bursts_col])
            ax_bursts_vs_egaba.annotate(
                f"$R^2$ = {r.rvalue ** 2:.2f} (p = {r.pvalue:.2g})",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                fontsize="xx-small",
                ha="right",
                va="top",
                color="k",
                # arrowprops=dict(arrowstyle='-|>',connectionstyle="arc3, rad=-0.1")
            )

        if bursts_vs_ggaba:
            ax_bursts_vs_ggaba = axs[1, int(bursts_vs_egaba)]
            sns.lineplot(
                data=df_sub_bursts,
                y=self.num_bursts_col,
                x=text.G_GABA,
                hue=text.TAU_KCC2,
                palette=COLOR.TAU_SM.get_cmap(),
                hue_norm=COLOR.TAU_SM.norm,
                ax=ax_bursts_vs_ggaba,
                marker=".",
                # multi="stack",
            )

            # regression
            reg_data = df_sub_bursts[[text.G_GABA, self.num_bursts_col]].dropna()
            sns.regplot(
                data=reg_data,
                y=self.num_bursts_col,
                x=text.G_GABA,
                ax=ax_bursts_vs_ggaba,
                scatter=False,
                logx=True,
                color="k",
            )
            ax_bursts_vs_ggaba.set_xscale("log")
            r = stats.linregress(reg_data[text.G_GABA], reg_data[self.num_bursts_col])
            print(r)
            ax_bursts_vs_ggaba.annotate(
                f"$R^2$ = {r.rvalue ** 2:.2f} (p = {r.pvalue:.2g})",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                fontsize="xx-small",
                ha="right",
                va="top",
            )

        if bursts_vs_tau_kcc2:
            ax_bursts_vs_tau = axs[1, int(bursts_vs_egaba + bursts_vs_ggaba)]

            sns.lineplot(
                data=df_sub_bursts,
                y=self.num_bursts_col,
                x=text.TAU_KCC2,
                hue=text.G_GABA,
                ax=ax_bursts_vs_tau,
                palette=COLOR.G_GABA_SM.cmap,
                hue_norm=COLOR.G_GABA_SM.norm,
                marker=".",
            )
            # reg
            reg_data = df_sub_bursts[[text.TAU_KCC2, self.num_bursts_col]].dropna()
            sns.regplot(
                data=reg_data,
                y=self.num_bursts_col,
                x=text.TAU_KCC2,
                ax=ax_bursts_vs_tau,
                scatter=False,
                logx=True,
                color="k",
            )
            r = stats.linregress(reg_data[text.TAU_KCC2], reg_data[self.num_bursts_col])
            print(r)
            ax_bursts_vs_tau.annotate(
                f"$R^2$ = {r.rvalue ** 2:.2f} (p = {r.pvalue:.2g})",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                fontsize="xx-small",
                ha="right",
                va="top",
            )

            ax_bursts_vs_tau.set_xscale("log")
            ax_bursts_vs_tau.set_xticks(taus, labels=[], minor=True)
            ax_bursts_vs_tau.set_xticks(taus[::2], labels=tau_labels[::2])
            ax_bursts_vs_tau.set_ylim(top=15)

        return fig


if __name__ == "__main__":
    # extend tau_KCC2 list to lower values, at the same ratio as existing values
    tau_KCC2_list = list(settings.TAU_KCC2_LIST)

    # add some more lower values for tau
    ratio = tau_KCC2_list[1] / tau_KCC2_list[0]
    # TODO: use proper ratio
    # ratio above ratio slightly off but results already cached.
    # ratio = np.sqrt(2)
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list
    tau_KCC2_list = [np.round(tau_KCC2_list[0] / ratio, 1)] + tau_KCC2_list

    self = Gve(
        seeds=(
            None,
            1234,
            5678,
            1426987,
            86751,
            16928,
            98766,
            876125,
            127658,
            9876,
            1010,
            876,
            12576,
            9681,
            814265,
        )[:10],
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
    self.run(time_per_value=60, EGABA_0=-80, EGABA_end=-38, mv_step=2)
    self.process()
    egabas = [-80, -72, -66, -60, -56, -54, -52, -50, -48, -40]

    self.plot(egabas=egabas, i_metric="diagram")
    self.save_figure(figs=self.figs, close=True)
