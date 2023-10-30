from itertools import product
import time
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cbook import flatten
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import settings
from core.analysis import burst_stats, ecl_to_egaba
from core.lrdfigure import MultiRunFigure
from settings import text, logging, time_unit
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
class Chloride(MultiRunFigure):
    fig_name = "figure_3_chloride"

    monitors = {
        "r_all": True,
        "sp_all": False,
        "state_mon": ["E_GABA"],
        "synapse_mon": False,
    }
    ignore = []

    def __init__(
        self,
        tau_KCC2s=settings.TAU_KCC2_LIST[::2],
        E_Cl_0s=(-60, -88),
        g_GABAs=(50, 25, 100),
        **kwargs,
    ):
        super().__init__(
            OrderedDict(
                g_GABA_max={"range": g_GABAs, "title": text.G_GABA},
                tau_KCC2_E={"range": tau_KCC2s, "title": text.TAU_KCC2},
                tau_KCC2_I=text.TAU_KCC2,  # same as E
                E_Cl_0={"range": E_Cl_0s, "title": text.ECL0},
            ),
            default_params=dict(dyn_cl=True),
            **kwargs,
        )

        self.tau_KCC2s = tau_KCC2s
        self.E_Cl_0s = E_Cl_0s
        self.g_GABAs = g_GABAs

        self.df_num_bursts = None

    def process(self, burst_window=120, **kwargs):
        T = np.round(self.df.index.values[-1])
        self.bin_size = bin_size = burst_window
        bins = np.arange(0, T + bin_size, bin_size)

        self.num_bursts_col = num_bursts_col = f"Number of bursts\n(per {bin_size} s)"
        logger.info(f"processing {num_bursts_col}".replace("\n", " "))
        self.all_rates = all_rates = self.df.xs(
            key="r_all", level="var", axis=1, drop_level=True
        )
        e_gaba_all = self.df.xs(key="E_GABA_all", level="var", axis=1, drop_level=True)
        # data structures
        df_bursts_long = pd.DataFrame(
            columns=[
                "run_idx",
                "E_Cl",
                "g_GABA",
                "KCC2",
                "Burst start time (s)",
                "Burst end time (s)",
            ]
        )
        df_egaba_long = pd.DataFrame(
            columns=["run_idx", "E_Cl", "g_GABA", "KCC2", "EGABA_0", "EGABA_end"]
        )
        for E_Cl_0, g_GABA, tau_KCC2 in product(
            self.E_Cl_0s, self.g_GABAs, self.tau_KCC2s
        ):
            for run_idx in range(len(self.seeds)):
                pop_rate = all_rates[g_GABA, tau_KCC2, E_Cl_0, run_idx]
                pop_egaba = e_gaba_all[g_GABA, tau_KCC2, E_Cl_0, run_idx]
                df_egaba_long = pd.concat(
                    [
                        df_egaba_long,
                        pd.DataFrame(
                            {
                                "run_idx": run_idx,
                                "E_Cl": E_Cl_0,
                                "g_GABA": g_GABA,
                                "KCC2": tau_KCC2,
                                "EGABA_0": pop_egaba.iloc[0],
                                "EGABA_end": pop_egaba.iloc[-1],
                                "EGABA_mean_end": pop_egaba.loc[T - bin_size :].mean(),
                            },
                            index=[df_egaba_long.shape[0]],
                        ),
                    ],
                    axis=0,
                )
                burst_start_ts, burst_end_ts = burst_stats(
                    pop_rate,
                    rate_std_thresh=2,
                    time_unit=time_unit,
                    plot_fig=False,
                )
                # store bursts
                for start_t, end_t in zip(burst_start_ts, burst_end_ts):
                    df_bursts_long.loc[df_bursts_long.shape[0]] = [
                        run_idx,
                        E_Cl_0,
                        g_GABA,
                        tau_KCC2,
                        start_t,
                        end_t,
                    ]

        df_bursts_long["bin"] = pd.cut(
            df_bursts_long["Burst start time (s)"],
            bins=np.append(bins, bins[-1] + bin_size),
            labels=bins.astype(int),
        )
        df_num_bursts = (
            df_bursts_long.groupby(["E_Cl", "g_GABA", "KCC2", "bin", "run_idx"])
            .count()
            .reset_index()
            .rename(columns={"Burst start time (s)": num_bursts_col})
            .fillna(0)
        )

        df_num_bursts["KCC2 g_GABA"] = (
            df_num_bursts["KCC2"].astype(int).map(str)
            + " "
            + df_num_bursts["g_GABA"].astype(int).map(str)
        )

        self.df_bursts_long = df_bursts_long
        self.df_num_bursts = df_num_bursts
        self.df_egaba_long = df_egaba_long
        return self

    def plot(self, default_tau=60, stripplot_alpha=0.5, stripplot_size=3, stripplot_jitter=0.4, **kwargs):
        if self.df_num_bursts is None:
            self.process(**kwargs)
        df_bursts_long = self.df_bursts_long
        df_num_bursts = self.df_num_bursts
        df_egaba = self.df_egaba_long

        # choose defaults for variables when keeping them fixed
        ecl_0 = self.E_Cl_0s[np.argmin(self.E_Cl_0s)]
        g_gaba = self.g_GABAs[0]
        assert default_tau in self.tau_KCC2s

        bins = sorted(df_num_bursts["bin"].unique())
        mid_bin = bins[len(bins) // 2]

        if (df_num_bursts["KCC2"] == df_num_bursts["KCC2"].round(0)).all():
            df_num_bursts["KCC2"] = df_num_bursts["KCC2"].astype(int)

        # colors
        lighten_g = np.linspace(0.6, 1.3, len(self.g_GABAs))

        cp = [
            COLOR.TAU_PAL_DICT[tau]
            if tau in COLOR.TAU_PAL_DICT
            else COLOR.TAU_SM.to_rgba(tau)
            for tau in self.tau_KCC2s
        ]
        cs = [lighten_color(c, light) for c in cp for light in lighten_g]
        cs_arr = np.array(cs).reshape(len(self.tau_KCC2s), len(self.g_GABAs), 3)

        # figure
        fig, axes = plt.subplot_mosaic(
            [
                ["schematic", "traces", "traces - zoom"],
                ["vary kcc2 - bursts", "vary kcc2 - bursts", "vary kcc2 - egaba"],
                ["."] * 3,
                ["traces - ecl", "traces - ecl", "traces - ecl - zoom"],
                ["vary ecl0 - bursts", "vary ecl0 - bursts", "vary ecl0 - bursts"],
                ["."] * 3,
                ["traces - ggaba", "traces - ggaba", "traces - ggaba"],
                ["vary ggaba - bursts", "vary ggaba - bursts", "vary ggaba - egaba"],
            ],
            gridspec_kw={
                "height_ratios": [0.2, 1, 0, 0.1, 0.6, 0, 0.15, 1.4],
                "width_ratios": [0.2, 1, 1],
                "hspace": 0.4,
                "wspace": 0.55,
                "top": 0.99,
                "bottom": 0.05,
                "left": 0.1,
                "right": 0.95,
            },
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL_no_cap),
        )
        self.fig, self.axs = fig, axes

        # letter_axes(axes.values())
        # schematic
        axes["schematic"].axis("off")
        axes["schematic"].annotate(
            "SCHEMATIC",
            xy=(0, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize="small",
        )

        ########################################
        # TRACES
        ########################################
        rate = self.df.xs(key="r_all", level="var", axis=1, drop_level=True)
        egaba = self.df.xs(key="E_GABA_all", level="var", axis=1, drop_level=True)
        bin_size_idx = (rate.index == self.bin_size).argmax()
        common_line_kwargs = {"lw": 0.5}
        subset_ranges = [None, np.r_[-bin_size_idx:0]]

        for ax_key, subset_range in zip(["traces", "traces - zoom"], subset_ranges):
            gs_traces = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=axes[ax_key], hspace=0.1, wspace=0.1
            )
            axes[ax_key].axis("off")
            axes["trace - rate"] = fig.add_subplot(gs_traces[0])
            axes["trace - egaba"] = fig.add_subplot(
                gs_traces[1], sharex=axes["trace - rate"]
            )

            if subset_range is not None:
                rate = rate.iloc[subset_range].reset_index(drop=True)
                egaba = egaba.iloc[subset_range].reset_index(drop=True)

            # trace - rate

            sns.lineplot(
                rate[g_gaba, default_tau, ecl_0, 0],
                ax=axes["trace - rate"],
                c=COLOR.K,
                **common_line_kwargs,
            )
            axes["trace - rate"].set_ylabel("population\nrate (Hz)")
            axes["trace - rate"].set_xlabel("")
            if subset_range is None:
                axes["trace - rate"].set_xlim(0, rate.index.values[-1])
            axes["trace - rate"].set_ylim(top=rate.values.max())
            sb = use_scalebar(
                axes["trace - rate"],
                # matchy=False,
                # sizey=20,
                matchx=False,
                sizex=0,
                hidex=True,
                hidey=True,
                loc="center left",
                bbox_to_anchor=(-0.25, 0.5),
                textprops={
                    "color": COLOR.K,
                    "ha": "right",
                    "y_rotation": 0,
                    "fontsize": "small",
                },
                fmt=".0f",
                labely="Hz",
            )

            # trace - egaba
            sns.lineplot(
                egaba[g_gaba, default_tau, ecl_0, 0],
                ax=axes["trace - egaba"],
                c=COLOR.K,
                **common_line_kwargs,
            )
            axes["trace - egaba"].set_ylabel(f"{text.EGABA}\n(mV)")
            axes["trace - egaba"].set_xlabel("")
            if subset_range is None:
                axes["trace - egaba"].set_xlim(0, egaba.index.values[-1])
            elif 0 not in subset_range:
                axes["trace - egaba"].set_xlim(
                    egaba.index.values[0], egaba.index.values[-1]
                )

            sb_egaba = use_scalebar(
                axes["trace - egaba"],
                # matchy=False,
                sizey=1,
                matchx=False,
                sizex=0,
                hidex=True,
                hidey=True,
                loc="center left",
                bbox_to_anchor=(-0.25, 0.5),
                textprops={
                    "color": COLOR.K,
                    "ha": "right",
                    "y_rotation": 0,
                    "fontsize": "small",
                },
                fmt=".0f",
                labely="mV",
            )

            sb_time = use_scalebar(
                axes["trace - egaba"],
                matchy=False,
                sizey=0,
                matchx=False,
                sizex=bin_size_idx if subset_range is not None else self.bin_size,
                hidex=True,
                hidey=True,
                loc="upper left",
                bbox_to_anchor=(0, 0.0),
                textprops={"fontsize": "small"},
                fmt=".0f",
                labelx="s",
            )

            if subset_range is None:
                # add shaded rectangle at end to show bin size
                for ax in [axes["trace - rate"], axes["trace - egaba"]]:
                    ax.axvspan(
                        rate.index.values[-1] - self.bin_size,
                        rate.index.values[-1],
                        color="k",
                        edgecolor="None",
                        alpha=0.1,
                        zorder=-10,
                    )

            else:
                # replace sb_time text with bin size
                sb_time.xlabel.set_text(f"{self.bin_size} s")
                if 0 in subset_range:
                    # CREATE a "break" in the x-axis at bin_size_idx
                    for ax in [axes["trace - rate"], axes["trace - egaba"]]:
                        ax.axvline(
                            bin_size_idx,
                            c="k",
                            lw=1,
                            zorder=100,
                            clip_on=False,
                            linestyle="-",
                        )

        ########################################
        # VARY KCC2
        ########################################
        # last bin (max time - burst_window to end)
        max_bin = df_num_bursts["bin"].max() - self.bin_size

        # vary kcc2 - bursts

        df_vary_kcc2_bursts = df_num_bursts[
            (df_num_bursts["E_Cl"] == ecl_0)
            & (df_num_bursts["g_GABA"] == g_gaba)
            # & (df_num_bursts["bin"] >= mid_bin)
            & (df_num_bursts["bin"] == max_bin)
        ]
        df_vary_kcc2_egaba = df_egaba[
            (df_egaba["E_Cl"] == ecl_0) & (df_egaba["g_GABA"] == g_gaba)
        ]

        sns.barplot(
            x="KCC2",
            y=self.num_bursts_col,
            order=self.tau_KCC2s,
            # hue="KCC2", palette=COLOR.TAU_PAL,
            dodge=False,
            color=COLOR.K,
            errorbar="se",
            # errwidth=0.2,
            # capsize=0.05,
            zorder=3,  # one tail of errorbar
            data=df_vary_kcc2_bursts,
            ax=axes["vary kcc2 - bursts"],
        )
        sns.stripplot(
            x="KCC2",
            y=self.num_bursts_col,
            order=self.tau_KCC2s,
            dodge=False,
            # color=".5",
            edgecolor="w",
            color=COLOR.K,
            # hue="KCC2", palette=COLOR.TAU_PAL,
            linewidth=1,
            zorder=6,
            size=stripplot_size,
            alpha=stripplot_alpha,
            jitter=stripplot_jitter,
            data=df_vary_kcc2_bursts[df_vary_kcc2_bursts[self.num_bursts_col] > 0],
            ax=axes["vary kcc2 - bursts"],
        )
        # vary kcc2 - egaba
        sns.violinplot(
            x="KCC2",
            y="EGABA_mean_end",
            # hue="KCC2",
            dodge=False,
            # palette=COLOR.TAU_PAL,
            color=COLOR.K,
            showfliers=False,
            saturation=1,
            linewidth=1,
            cut=0,
            data=df_vary_kcc2_egaba,
            ax=axes["vary kcc2 - egaba"],
        )
        # sns.stripplot(
        #     x="KCC2",
        #     y="EGABA_end",
        #     # hue="KCC2",
        #     dodge=False,
        #     # palette=COLOR.TAU_PAL,
        #     color=COLOR.K,
        #     # size=4,
        #     linewidth=1,
        #     # color=COLOR.Pu,
        #     edgecolor="w",
        #     alpha=stripplot_alpha,
        #     data=df_vary_kcc2_egaba,
        #     ax=axes["vary kcc2 - egaba"],
        # )

        axes["vary kcc2 - egaba"].set_ylabel(f"{text.EGABA} (mV)\nat end of run ")
        for ax in [axes["vary kcc2 - bursts"], axes["vary kcc2 - egaba"]]:
            ax.set(xlabel=f"{text.TAU_KCC2} (s)")
        axes["vary kcc2 - bursts"].annotate(
            f"{text.G_GABA} = {g_gaba} nS",
            xy=(0.1, 0.9),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
            color=COLOR.G_GABA_SM.to_rgba(g_gaba),
        )

        # remove legends
        for ax in [axes["vary kcc2 - bursts"], axes["vary kcc2 - egaba"]]:
            try:
                ax.get_legend().remove()
            except AttributeError:
                # if hue wasn't provided
                pass

        ########################################
        # VARY ECL0 - traces
        # Have 2 traces per axes - one for each E_Cl_0
        ########################################

        egaba_start_values = [
            ecl_to_egaba(ecl_0_plot) for ecl_0_plot in sorted(self.E_Cl_0s)
        ]
        EGABA_0_PALETTE = [
            COLOR.EGABA_SM.to_rgba(v) for v in egaba_start_values
        ]

        # vary ecl0 - traces
        # re-init
        egaba = self.df.xs(key="E_GABA_all", level="var", axis=1, drop_level=True)
        axes_ecl = [axes["traces - ecl"], axes["traces - ecl - zoom"]]
        for ax_ecl, subset_range in zip(axes_ecl, subset_ranges):
            for i, ecl_0_plot in enumerate(self.E_Cl_0s):
                egaba_start_value = ecl_to_egaba(ecl_0_plot)
                if subset_range is not None:
                    egaba = egaba.iloc[subset_range].reset_index(drop=True)
                sns.lineplot(
                    egaba[g_gaba, default_tau, ecl_0_plot, 0],
                    ax=ax_ecl,
                    c=COLOR.EGABA_SM.to_rgba(egaba_start_value),
                    ls="--" if i > 0 else "-",
                    **common_line_kwargs,
                )
                # annotate starting point with E_Cl_0 for non-zoom
                if subset_range is None:
                    ax_ecl.annotate(
                        f"{egaba_start_value:>.0f}",
                        xy=(0, egaba_start_value),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center_baseline",
                        fontsize="small",
                        c=COLOR.EGABA_SM.to_rgba(egaba_start_value),
                    )

            if subset_range is None:
                ax_ecl.set_xlim(0, egaba.index.values[-1])
            elif 0 not in subset_range:
                ax_ecl.set_xlim(egaba.index.values[0], egaba.index.values[-1])

            sb_ecl = use_scalebar(
                ax_ecl,
                matchx=False,
                sizex=self.bin_size if subset_range is None else bin_size_idx,
                hidex=True,
                hidey=True,
                loc="center left",
                bbox_to_anchor=(0.25, 0.0) if subset_range is None else (-0.35, -0.1),
                textprops={
                    "color": COLOR.EGABA,
                    # "ha": "right",
                    "va": "center",
                    "y_rotation": 0,
                    "fontsize": "small",
                },
                fmt=".0f" if subset_range is None else ".2f",
                labely="mV",
                labelx="s",
            )
            if subset_range is None:
                # shade in last bin
                ax_ecl.axvspan(
                    egaba.index.values[-1] - self.bin_size,
                    egaba.index.values[-1],
                    color="k",
                    edgecolor="None",
                    alpha=0.1,
                    zorder=-10,
                )
                ax_ecl.annotate(
                    f"{text.EGABA}\n(mV)",
                    xy=(0, 0.5),
                    xytext=(-30, 0),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize="small",
                    rotation=90,
                    clip_on=False,
                    # annotation_clip=False,
                    color=COLOR.EGABA,
                )
            else:
                # replace sb_time text with bin size (instead of bin_size_idx)
                sb_ecl.xlabel.set_text(f"{self.bin_size} s")

        ########################################
        # VARY ECL0 - bursts
        ########################################
        # vary ecl0 - bursts

        df_vary_ecl0_bursts = df_num_bursts[
            (df_num_bursts["g_GABA"] == g_gaba) & (df_num_bursts["bin"] >= mid_bin)
        ]
        EGABA_0 = f"${text.EGABA}_0$"
        df_vary_ecl0_bursts[EGABA_0] = df_vary_ecl0_bursts["E_Cl"].apply(
            lambda x: ecl_to_egaba(x)
        )
        egaba0_order = sorted(df_vary_ecl0_bursts[EGABA_0].unique())
        sns.barplot(
            x="KCC2",
            y=self.num_bursts_col,
            hue=EGABA_0,
            hue_order=egaba0_order,
            palette=EGABA_0_PALETTE,
            # color=COLOR.NUM_BURSTS_COLOR,
            errorbar="se",
            # errwidth=0.2,
            # capsize=0.05,
            zorder=5,  # one tail of errorbar
            data=df_vary_ecl0_bursts,
            ax=axes["vary ecl0 - bursts"],
        )
        sns.stripplot(
            x="KCC2",
            order=self.tau_KCC2s,
            y=self.num_bursts_col,
            hue=EGABA_0,
            hue_order=egaba0_order,
            palette=EGABA_0_PALETTE,
            color=COLOR.NUM_BURSTS_COLOR,
            dodge=True,
            edgecolor="w",
            linewidth=1,
            size=stripplot_size,
            zorder=6,
            alpha=stripplot_alpha,
            legend=False,
            jitter=stripplot_jitter,
            data=df_vary_ecl0_bursts[df_vary_ecl0_bursts[self.num_bursts_col] > 0],
            ax=axes["vary ecl0 - bursts"],
        )

        axes["vary ecl0 - bursts"].legend(
            frameon=False, fontsize="small", title=EGABA_0, title_fontsize="small"
        )
        axes["vary ecl0 - bursts"].set_xlabel(f"{text.TAU_KCC2} (s)")

        ########################################
        # VARY G_GABA
        ########################################

        # vary ggaba - traces
        # re-init
        rates = self.df.xs(key="r_all", level="var", axis=1, drop_level=True)
        egaba = self.df.xs(key="E_GABA_all", level="var", axis=1, drop_level=True)

        ggaba_subset_ranges = subset_ranges[:1]
        axes_ggaba = [axes["traces - ggaba"]]
        for ax_ggaba, subset_range in zip(axes_ggaba, ggaba_subset_ranges):
            for i, g_gaba_plot in enumerate(self.g_GABAs):
                if subset_range is not None:
                    rates = rates.iloc[subset_range].reset_index(drop=True)
                    egaba = egaba.iloc[subset_range].reset_index(drop=True)
                egaba_s = egaba[g_gaba_plot, default_tau, self.E_Cl_0s[0], 0]
                sns.lineplot(
                    egaba_s,
                    ax=ax_ggaba,
                    c=COLOR.G_GABA_SM.to_rgba(g_gaba_plot),
                    label=f"{g_gaba_plot}",
                    legend=False,
                    **common_line_kwargs,
                )
            if subset_range is None:
                ax_ggaba.set_xlim(0, egaba.index.values[-1])
                ax_ggaba.legend(
                    loc="center right",
                    bbox_to_anchor=(1.0, 0.0),
                    frameon=False,
                    labelcolor="linecolor",
                    labelspacing=0,
                    handlelength=0,
                    handletextpad=0,
                    borderaxespad=0,
                    borderpad=0,
                    title=f"{text.G_GABA} (nS)",
                    title_fontsize="small",
                    fontsize="small",
                    ncol=len(self.g_GABAs),
                )
            elif 0 not in subset_range:
                ax_ggaba.set_xlim(egaba.index.values[0], egaba.index.values[-1])

            sb_ggaba = use_scalebar(
                ax_ggaba,
                matchx=False,
                sizex=self.bin_size if subset_range is None else bin_size_idx,
                hidex=True,
                hidey=True,
                loc="center left",
                bbox_to_anchor=(0.25, 0.0) if subset_range is None else (-0.35, -0.1),
                textprops={
                    "color": COLOR.EGABA,
                    # "ha": "right",
                    "va": "center",
                    "y_rotation": 0,
                    "fontsize": "small",
                },
                fmt=".0f" if subset_range is None else ".2f",
                labely="mV",
                labelx="s",
            )
            if subset_range is None:
                if len(ggaba_subset_ranges) > 1:
                    # shade in last bin
                    ax_ggaba.axvspan(
                        egaba.index.values[-1] - self.bin_size,
                        egaba.index.values[-1],
                        color="k",
                        edgecolor="None",
                        alpha=0.1,
                        zorder=-10,
                    )
                ax_ggaba.annotate(
                    f"{text.EGABA}\n(mV)",
                    xy=(0, 0.5),
                    xytext=(-30, 0),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize="small",
                    rotation=90,
                    clip_on=False,
                    # annotation_clip=False,
                    color=COLOR.EGABA,
                )
            else:
                # replace sb_time text with bin size (instead of bin_size_idx)
                sb_ggaba.xlabel.set_text(f"{self.bin_size} s")

        # vary ggaba - bursts
        df_vary_ggaba_bursts = df_num_bursts[
            (df_num_bursts["E_Cl"] == ecl_0) & (df_num_bursts["bin"] >= mid_bin)
        ]
        df_vary_ggaba_bursts[text.G_GABA] = df_vary_ggaba_bursts["g_GABA"].astype(int)

        df_vary_ggaba_egaba = df_egaba[(df_egaba["E_Cl"] == ecl_0)]
        df_vary_ggaba_egaba[text.G_GABA] = df_vary_ggaba_egaba["g_GABA"].astype(int)

        g_gaba_order = sorted(df_vary_ggaba_bursts[text.G_GABA].unique())
        g_gaba_palette = {
            g_gaba: COLOR.G_GABA_SM.to_rgba(g_gaba) for g_gaba in g_gaba_order
        }

        sns.barplot(
            x="KCC2",
            y=self.num_bursts_col,
            hue=text.G_GABA,
            hue_order=g_gaba_order,
            palette=g_gaba_palette,
            errorbar="se",
            zorder=5,  # one tail of errorbar
            data=df_vary_ggaba_bursts,
            ax=axes["vary ggaba - bursts"],
        )
        sns.stripplot(
            x="KCC2",
            order=self.tau_KCC2s,
            y=self.num_bursts_col,
            hue=text.G_GABA,
            hue_order=g_gaba_order,
            palette=g_gaba_palette,
            dodge=True,
            edgecolor="w",
            linewidth=1,
            zorder=6,
            size=stripplot_size,
            alpha=stripplot_alpha,
            jitter=stripplot_jitter,
            legend=False,
            data=df_vary_ggaba_bursts[df_vary_ggaba_bursts[self.num_bursts_col] > 0],
            ax=axes["vary ggaba - bursts"],
        )

        axes["vary ggaba - bursts"].set_xlabel(f"{text.TAU_KCC2} (s)")
        axes["vary ggaba - bursts"].set_ylim(bottom=0)
        axes["vary ggaba - bursts"].set_ylabel(self.num_bursts_col)
        axes["vary ggaba - bursts"].legend(
            frameon=False,
            fontsize="small",
            title=f"{text.G_GABA} (nS)",
            title_fontsize="small",
        )

        # vary ggaba - egaba
        sns.violinplot(
            x="KCC2",
            y="EGABA_mean_end",
            hue=text.G_GABA,
            hue_order=g_gaba_order,
            palette=g_gaba_palette,
            showfliers=False,
            data=df_vary_ggaba_egaba,
            # color='k',
            linewidth=0.8,
            saturation=1,
            width=0.9,
            cut=0,
            ax=axes["vary ggaba - egaba"],
        )
        # sns.stripplot(
        #     x="KCC2",
        #     y="EGABA_end",
        #     hue=text.G_GABA,
        #     hue_order=g_gaba_order,
        #     palette=g_gaba_palette,
        #     dodge=True,
        #     linewidth=1,
        #     edgecolor="w",
        #     alpha=stripplot_alpha,
        #     legend=False,
        #     data=df_vary_ggaba_egaba,
        #     ax=axes["vary ggaba - egaba"],
        # )
        axes["vary ggaba - egaba"].legend().remove()
        axes["vary ggaba - egaba"].set_xlabel(f"{text.TAU_KCC2} (s)")
        axes["vary ggaba - egaba"].set_ylabel(f"{text.EGABA} (mV)\nat end of run ")
        # add borders between ordinal x-axis values
        _vline_style = dict(
            color="k",
            ls="--",
            alpha=0.5,
            lw=0.5,
            zorder=100,
        )
        axes["vary ggaba - bursts"].vlines(
            np.arange(len(self.tau_KCC2s)) + 0.5,
            *axes["vary ggaba - bursts"].get_ylim(),
            **_vline_style,
        )
        axes["vary ggaba - egaba"].vlines(
            np.arange(len(self.tau_KCC2s)) + 0.5,
            *axes["vary ggaba - egaba"].get_ylim(),
            **_vline_style,
        )

        if "vary ggaba - egaba - alt" in axes:
            sns.violinplot(
                x=text.G_GABA,
                order=g_gaba_order,
                y="EGABA_mean_end",
                hue="KCC2",
                palette=COLOR.TAU_PAL_DICT,
                showfliers=False,
                data=df_vary_ggaba_egaba,
                # color='k',
                linewidth=0.8,
                saturation=1,
                width=0.9,
                cut=0,
                ax=axes["vary ggaba - egaba - alt"],
            )

    def plot_supp(
        self,
        timeit=True,
        burst_window=120,
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

        ncols = len(self.E_Cl_0s)
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

        lighten_g = np.linspace(0.6, 1.3, len(self.g_GABAs))

        cp = [
            COLOR.TAU_PAL_DICT[tau]
            if tau in COLOR.TAU_PAL_DICT
            else COLOR.TAU_SM.to_rgba(tau)
            for tau in self.tau_KCC2s
        ]
        cs = [lighten_color(c, light) for c in cp for light in lighten_g]
        cs_arr = np.array(cs).reshape(len(self.tau_KCC2s), len(self.g_GABAs), 3)

        # data structures
        df_long = pd.DataFrame(
            columns=[
                "run_idx",
                "E_Cl",
                "g_GABA",
                "KCC2",
                "Burst start time (s)",
                "Burst end time (s)",
            ]
        )

        y_spacing = 20
        off_i = 0
        all_lines = []
        for e, E_Cl_0 in enumerate(self.E_Cl_0s):
            bursts = []
            tau_bursts = defaultdict(
                list
            )  # create dict of list objects where key=tau, list = bursts
            all_lines = []
            for g, g_GABA in enumerate(self.g_GABAs):
                gs_g = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[g, e], hspace=0.05)
                axs[g * 2, e] = ax_gaba = fig.add_subplot(
                    gs_g[0], sharey=ax_gaba, sharex=ax_r
                )
                axs[g * 2 + 1, e] = ax_r = fig.add_subplot(
                    gs_g[1], sharey=ax_r, sharex=ax_r
                )

                lines = []
                line_kwargs = dict(alpha=1, linewidth=0.5)
                d = {}  # store egabas
                d_e = {}  # store egabas
                d_i = {}  # store egabas

                # plot population rates
                for i, tau_KCC2 in enumerate(self.tau_KCC2s):
                    for run_idx in range(len(self.seeds)):
                        pop_rate = all_rates[g_GABA, tau_KCC2, E_Cl_0, run_idx]
                        if run_idx == 0:
                            logger.debug(
                                f"plotting for g={g_GABA} tau={tau_KCC2} s with E_Cl_0={E_Cl_0}"
                            )
                            d[f"t{tau_KCC2}"] = e_gaba_all[
                                g_GABA, tau_KCC2, E_Cl_0, run_idx
                            ]
                            d_e[f"t{tau_KCC2}"] = e_gaba_E[
                                g_GABA, tau_KCC2, E_Cl_0, run_idx
                            ]
                            d_i[f"t{tau_KCC2}"] = e_gaba_I[
                                g_GABA, tau_KCC2, E_Cl_0, run_idx
                            ]
                            egaba = d[f"t{tau_KCC2}"].values
                            if e == 1:
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
                                    cmap=COLOR.EGABA_SM.get_cmap(),
                                    norm=COLOR.EGABA_SM.norm,
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
                                    start_t,
                                    end_t,
                                ]

                ax_r.set_xlim(0, T)
                if g == 0 and e == 0:
                    ax_r.set_xticks(np.arange(0, T + bin_size, bin_size))
                ax_r.set_ylim(0, rmax + y_spacing * (i + off_i))

                # adjust_spines(ax_r, ['left', 'bottom'], 0)
                # ax_r.spines['left'].set_alpha(0.25)
                # ax_r.spines['right'].set_linestyle(':')
                # if e == 0:
                #     ax_r.set_ylabel(f"{text.G_GABA} = {g_GABA}\npopulation rate (Hz)")
                # else:
                #   ax_r.set_ylabel(f"population rate (Hz)")
                # ax_gaba = ax_r.twinx()
                # adjust_spines(ax_r, [], 0)

                ax_r.grid(True, "major", "x", zorder=-len(self.tau_KCC2s) * 10)

                # ax_gaba = ax_r
                # ax_gaba.spines['left'].set_linewidth(3)
                # ax_gaba.spines['left'].set_alpha(0.1)
                if g == 0:
                    egaba_start_value = 0.8 * E_Cl_0 + 0.2 * -18
                    ax_gaba.annotate(
                        f"{text.EGABA} = {egaba_start_value:.1f} mV",
                        xy=(0, egaba_start_value),
                        fontsize="large",
                        bbox=dict(boxstyle="round", fc="w", ec="w", pad=0.01),
                        xytext=(T / 4, e_gaba_all.values.max()),
                        va="bottom",
                        color=COLOR.EGABA_SM.to_rgba(egaba_start_value),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            connectionstyle="arc3,rad=0.2" if e == 0 else None,
                            color=COLOR.EGABA_SM.to_rgba(egaba_start_value),
                            ec="k",
                        ),
                    )
                    if e > 0:
                        adjust_spines(ax_r, [], 0)
                else:
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
                if g == 0:
                    ax_gaba.set_yticklabels(yticks[1::2])
                ax_gaba.set_xlim([0, T])
                ax_gaba.grid(True, "both", "both", zorder=-99)

                if e == 0:
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
                        bbox_to_anchor=(-0.1, 0.1),
                    )
                    sb.ylabel._text.set_fontsize("small")
                    sb.ylabel._text.set_rotation(0)
                    sb.ylabel.set_text("10 Hz")
                    sb = use_scalebar(
                        ax_r,
                        matchy=False,
                        sizey=0,
                        matchx=False,
                        sizex=burst_window,
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
                    ax_gaba.set_ylabel(f"{text.EGABA}\n(mV)")
                    # c = lighten_color(COLOR.K, lighten_g[g])
                    c = COLOR.G_GABA_SM.to_rgba(g_GABA)
                    ax_gaba.annotate(
                        f"{text.G_GABA}\n{g_GABA} nS",
                        xy=(-0.05, 1.05),
                        xycoords="axes fraction",
                        fontsize="medium",
                        ha="center",
                        va="bottom",
                        c=c,
                    )

                else:
                    ax_gaba.set_ylabel("")
                    c = lighten_color(COLOR.K, lighten_g[g])
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

                if colorbar:
                    logger.debug(f"creating colorbar for E_Cl_0={E_Cl_0}")
                    from matplotlib.colorbar import Colorbar

                    cb: Colorbar = colorbar_inset(COLOR.EGABA_SM, ax=ax_r)
                    cb.set_label(ax_gaba.get_ylabel())
                    ax_gaba.set_ylabel("")
                    ax_gaba.set_ylim(COLOR.EGABA_SM.get_clim())
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
            df_long.groupby(["E_Cl", "g_GABA", "KCC2", "bin", "run_idx"])
            .count()
            .reset_index()
            .rename(columns={"Burst start time (s)": num_bursts_col})
            .fillna(0)
        )

        df_num_bursts["KCC2 g_GABA"] = (
            df_num_bursts["KCC2"].astype(int).map(str)
            + " "
            + df_num_bursts["g_GABA"].astype(int).map(str)
        )
        share_lims = None
        for e, ecl in enumerate(self.E_Cl_0s):
            axs[-1, e] = share_lims = fig.add_subplot(
                gs[-1, e], sharex=share_lims, sharey=share_lims
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
                data=df_num_bursts[df_num_bursts["E_Cl"] == ecl],
                ax=axs[-1, e],
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
            axs[-1, e].set_xticks(np.arange(len(bins), dtype=int) - shift)
            axs[-1, e].set_yticks(np.arange(0, axs[-1, e].get_ylim()[1], 2, dtype=int))
            axs[-1, e].set_yticks(
                np.arange(0, axs[-1, e].get_ylim()[1], 1, dtype=int), minor=True
            )
            axs[-1, e].grid(True, axis="both", which="both", alpha=0.4, zorder=-99)
            axs[-1, e].set_xlim(-shift, len(bins) - 1 - shift)
            if e == 0:
                axs[-1, e].legend().remove()
            else:
                axs[-1, e].set_ylabel("")

        axs[-1, 0].set_ylim(0)

        tau_kcc2s_leg_v = [None] * len(self.tau_KCC2s) * (len(self.g_GABAs) - 1) + [
            (f"{tau_KCC2}") for tau_KCC2 in self.tau_KCC2s
        ]
        g_gaba_str = " ".join([f"{g:.0f}" for g in self.g_GABAs])
        leg = axs[-1, 0].legend(
            all_lines,
            tau_kcc2s_leg_v,
            loc="upper left",
            bbox_to_anchor=(-0.0, 1),
            ncol=len(self.g_GABAs),
            columnspacing=-2.5,
            handlelength=1,
            handleheight=1,
            handletextpad=3.8,
            labelspacing=0.3,
            borderaxespad=0.0,
            fontsize="small",
            frameon=True,
            facecolor="w",
            edgecolor="None",
            title=f"{text.G_GABA} (nS)\n{g_gaba_str}  {text.TAU_KCC2} (s)   ",
            title_fontsize="small",
        )

        for txt in leg.get_texts():
            txt.set_ha("left")

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

        for _ax in axs[::2, 0]:
            letter_axes(
                _ax,
                xy=(0, _ax.get_position().y0),
                xycoords="figure fraction",
                ha="left",
                va="bottom",
            )

        for ax_i in axs[-1, :]:
            ax_i.set_xlabel(f"{text.TIME} bin" + " (%s)" % time_unit)
            ax_i.set_xticklabels(
                [f"{t}" for t in np.arange(0, T + bin_size, bin_size, dtype=int)]
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
    cl = Chloride(seeds=(None, 1038, 1337, 1111, 1010, 1011, 1101, 1110, 11110, 111100))
    cl.run(duration=600)
    cl.plot(timeit=True, colorbar=False)
    if settings.SAVE_FIGURES:
        cl.save_figure(use_args=False, close=False)
    plt.show()
