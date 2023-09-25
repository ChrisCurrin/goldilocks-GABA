import itertools
import time
from collections import OrderedDict
from matplotlib import patheffects
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from core.analysis import burst_stats
from core.lrdfigure import MultiRunFigure
from settings import (
    text,
    logging,
    time_unit,
    PAGE_W_FULL,
    PAGE_H_half,
    COLOR,
)
from style.axes import colorline

logger = logging.getLogger(__name__)

round_EGABA = f"{text.E_GABA}"


class Params(MultiRunFigure):
    fig_name = "figure_s1_gampa"

    monitors = {
        "r_all": True,
        "sp_all": False,
        "state_mon": ["E_GABA", "g_GABA", "I_GABA_rec"],
        "synapse_mon": False,
    }

    def __init__(
        self,
        gGABAs=(50.0,),
        gAMPAs=(5.0,),
        gNMDAs=(5.0,),
        time_per_value=60,
        EGABA_0=-74,
        EGABA_end=-40,
        mv_step=2,
        **kwargs,
    ):
        self.gGABAs = gGABAs
        self.gAMPAs = gAMPAs
        self.gNMDAs = gNMDAs
        # remove 0 values
        self.gGABAs = tuple(g for g in self.gGABAs if g > 0)
        self.gAMPAs = tuple(g for g in self.gAMPAs if g > 0)
        self.gNMDAs = tuple(g for g in self.gNMDAs if g > 0)

        self.time_per_value = time_per_value
        self.EGABA_0 = EGABA_0
        self.EGABA_end = EGABA_end
        self.mv_step = mv_step
        self.figs = []

        ehco3 = -18
        phco3 = 0.2
        pcl = 1 - phco3
        diff = EGABA_end - EGABA_0
        values = diff // mv_step
        self.num_ecl_steps = num_ecl_steps = values - 1
        self.duration = duration = values * time_per_value
        ecl_0 = round((EGABA_0 - phco3 * ehco3) / pcl, 2)
        ecl_end = round((EGABA_end - phco3 * ehco3) / pcl, 2)

        kwargs["E_Cl_0"] = ecl_0
        kwargs["E_Cl_end"] = ecl_end

        manual_cl = kwargs.pop("manual_cl", True)

        super().__init__(
            OrderedDict(
                g_GABA_max={"range": self.gGABAs, "title": text.G_GABA},
                g_AMPA_max={"range": self.gAMPAs, "title": text.G_AMPA},
                g_NMDA_max={"range": self.gNMDAs, "title": text.G_NMDA},
            ),
            default_params=dict(
                manual_cl=manual_cl, duration=duration, num_ecl_steps=num_ecl_steps
            ),
            **kwargs,
        )

    def process(self):
        run_idxs = list(range(len(self.seeds)))
        T = np.round(self.df.index.values[-1])
        bin_size = self.time_per_value
        bins = np.arange(0, T, self.time_per_value)

        ###############
        # static EGABA
        ###############
        df_g_E = pd.DataFrame(
            columns=[
                "run_idx",
                text.EGABA,
                text.G_GABA,
                text.G_AMPA,
                text.G_NMDA,
                "Burst start time (s)",
            ]
        )

        for gGABA, gAMPA, gNMDA, run_idx in tqdm(
            list(itertools.product(self.gGABAs, self.gAMPAs, self.gNMDAs, run_idxs)),
            desc="processing bursts",
        ):
            df_egaba = self.df[gGABA, gAMPA, gNMDA, run_idx, "E_GABA_all"]
            df_rates = self.df[gGABA, gAMPA, gNMDA, run_idx, "r_all"]
            burst_start_ts, burst_end_ts = burst_stats(
                df_rates, rate_std_thresh=3, time_unit=time_unit, plot_fig=False
            )
            logger.debug(
                f"gGABA={gGABA}, gAMPA={gAMPA}, run_idx={run_idx} burst_start_ts={burst_start_ts}"
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
                    df_g_E.loc[df_g_E.shape[0]] = [
                        run_idx,
                        egaba,
                        gGABA,
                        gAMPA,
                        gNMDA,
                        start_t,
                    ]
                if len(burst_ts) == 0:
                    # add expected observations so there is always an entry
                    df_g_E.loc[df_g_E.shape[0]] = [
                        run_idx,
                        egaba,
                        gGABA,
                        gAMPA,
                        gNMDA,
                        np.nan,
                    ]

        # analysis
        # df_g_E["bin"] = pd.cut(
        #     df_g_E["Burst start time (s)"],
        #     bins=np.append(bins, bins[-1] + bin_size),
        #     labels=bins.astype(int),
        # )

        num_bursts_col = f"Number of bursts\n(per {bin_size} s)"
        df_g_E_bursts = (
            df_g_E.groupby(
                [
                    text.EGABA,
                    text.G_GABA,
                    text.G_AMPA,
                    text.G_NMDA,
                    "run_idx",
                ],
                as_index=False,
            )
            .count()
            .rename(columns={"Burst start time (s)": num_bursts_col})
        )
        # df_g_E_bursts[num_bursts_col] = df_g_E_bursts[num_bursts_col].fillna(0)/bin_size/len(self.seeds)
        df_g_E_bursts["hyperpolaring\nEGABA"] = pd.cut(
            df_g_E_bursts[text.EGABA],
            bins=[-100, -55, 0],
            labels=[True, False],
        )

        self.df_g_E = df_g_E
        self.df_g_E_bursts = df_g_E_bursts
        self.num_bursts_col = num_bursts_col

    def plot(self, timeit=True, **kwargs):
        super().plot(**kwargs)
        logger.info("plotting")
        plot_time_start = time.time()

        self.plot_gampa(**kwargs)

        self.plot_rate_example(**kwargs)

        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self

    def plot_rate_example(self, **kwargs):
        logger.info("plotting rate example")

        fig, axes = plt.subplot_mosaic(
            [[f"rates_color_{g}"] for g in self.gNMDAs],
            # subplot_kw={"sharex": "all", "sharey": "all"},
            figsize=(PAGE_W_FULL, PAGE_H_half),
            gridspec_kw={"hspace": 0.2},
        )
        gGABA = self.gGABAs[1]
        gAMPA = self.gAMPAs[0]

        run_idx = 0

        r_all = self.df.xs(key="r_all", level="var", axis=1, drop_level=True).values
        r_norm = norm = Normalize(r_all.min(), r_all.max())

        bins = np.arange(0, self.duration, self.time_per_value)
        for gNMDA in self.gNMDAs:
            ax = axes[f"rates_color_{gNMDA}"]
            df_egaba = self.df[gGABA, gAMPA, gNMDA, run_idx, "E_GABA_all"]
            df_rates = self.df[gGABA, gAMPA, gNMDA, run_idx, "r_all"]

            # plot rates with color as egaba
            vmin = df_egaba.values.min()
            vmax = df_egaba.values.max()
            norm = Normalize(vmin, vmax)

            colorline(
                df_rates.index,
                df_rates.values,
                z=df_egaba.values,
                cmap=COLOR.EGABA_SM.get_cmap(),
                norm=COLOR.EGABA_SM.norm,
                linewidth=0.5,
                ax=ax,
                rasterized=True,
            )

            ax.set_ylabel("Population rate (Hz)")
            if gNMDA == self.gNMDAs[-1]:
                ax.set_xlabel("Time (s)")
            ax.set_title(
                f"gGABA={gGABA}, gAMPA={gAMPA}, gNMDA={gNMDA}",
                fontsize="small",
                va="top",
            )

        # set x and y lims
        for ax_name, ax in axes.items():
            ax.set_xlim(0, self.duration)
            ax.set_ylim(0, r_norm.vmax)
            # add grid according to bins
            ax.set_xticks(bins, minor=True)
            ax.grid(True, axis="x", which="minor", alpha=0.5, linestyle="--")

        self.sim_name, prev_sim_name = f"{self.fig_name}_rates_example", self.sim_name
        self.save_figure(figs=[fig], use_args=True)
        self.sim_name = prev_sim_name

    def plot_gampa(
        self, axes=None, egaba_as_row=True, egabas=(-48, -56), fig_kwargs=None
    ):
        # check egaba in num_ecl_steps
        run_egaba = np.arange(self.EGABA_0, self.EGABA_end, self.mv_step).round(2)
        round_egabas = np.round(egabas, 2)
        assert np.all(
            np.isin(round_egabas, run_egaba)
        ), f"egabas={egabas} not all in run_egaba={run_egaba}"

        if axes is None:
            if fig_kwargs is None:
                fig_kwargs = {}
            gridspec_kw = fig_kwargs.pop("gridspec_kw", {"hspace": 0.7, "wspace": 0.1})

            fig, axes = plt.subplots(
                len(egabas) if egaba_as_row else 1,
                ncols=len(self.gNMDAs),
                squeeze=False,
                sharex=True,
                sharey=True,
                gridspec_kw=gridspec_kw,
                **fig_kwargs,
            )
            self.fig = fig
        # df_g_E = self.df_g_E
        df_g_E_bursts = self.df_g_E_bursts
        num_bursts_col = self.num_bursts_col

        df_g_E_bursts[text.G_AMPA] = df_g_E_bursts[text.G_AMPA].astype(int)
        df_g_E_bursts[text.G_GABA] = df_g_E_bursts[text.G_GABA].astype(int)
        df_g_E_bursts[text.G_NMDA] = df_g_E_bursts[text.G_NMDA].astype(float)

        # bursts v G with G_AMPA
        for (g, g_nmda), (e, egaba) in itertools.product(
            enumerate(sorted(self.gNMDAs)), enumerate(egabas)
        ):
            # color palette that os from purple to yellow
            if egaba >= -48:
                pal = "YlOrRd_r"
                errcolor = "r"
            elif -60 < egaba < -48:
                pal = sns.blend_palette(
                    ["purple", "pink"], len(self.gAMPAs), as_cmap=False
                )
                errcolor = "purple"
            else:
                pal = "YlGnBu_r"
                errcolor = "b"

            sns.barplot(
                x=text.G_GABA,
                y=num_bursts_col,
                # hue="AMPA+NMDA",
                # hue_order=order,
                hue=text.G_AMPA,
                hue_order=sorted(
                    df_g_E_bursts[text.G_AMPA].unique(), reverse=False
                ),
                # err_style='band',
                errcolor=errcolor,
                errwidth=0.5,
                orient="v",
                palette=pal,
                # palette=cs,
                data=df_g_E_bursts[
                    (df_g_E_bursts[text.EGABA] == egaba)
                    & (df_g_E_bursts[text.G_NMDA] == g_nmda)
                ],
                ax=axes[e, g],
            )
            # remove legend
            axes[e, g].get_legend().remove()

            # y label
            if g == 0:
                axes[e, g].set_ylabel(num_bursts_col)
                # annotate egaba
                axes[e, g].annotate(
                    f"{text.E_GABA}\n{egaba:.0f} mV",
                    xy=(0.0, 1.0),
                    xycoords="axes fraction",
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize="medium",
                    color=COLOR.EGABA_SM.to_rgba(egaba),
                    path_effects=[
                        patheffects.withSimplePatchShadow(offset=(0.5, -0.5))
                    ],
                )
            else:
                axes[e, g].set_ylabel("")

            # title
            if e == 0:
                axes[e, g].set_title(
                    f"{text.G_NMDA}\n{g_nmda} nS", va="bottom", fontsize="small"
                )

            # x label
            if e == len(axes) - 1:
                axes[e, g].set_xlabel(f"{text.G_GABA} (ns)")
            else:
                axes[e, g].set_xlabel("")

            # yticks and grid
            # using MaxNLocator to get nice tick spacing
            axes[e, g].yaxis.set_major_locator(MaxNLocator("auto", integer=True))
            axes[e, g].grid(True, axis="y", alpha=0.5, zorder=-1)

            if g == 0:
                leg = axes[e, g].legend(
                    ncol=len(df_g_E_bursts[text.G_AMPA].unique()),
                    loc="upper left",
                    bbox_to_anchor=(0, 1.0),
                    handlelength=1,
                    handletextpad=0,
                    columnspacing=0,
                    # mode="expand",
                    fontsize="x-small",
                    frameon=False,
                    title=f"{text.G_AMPA} (ns)",
                    title_fontsize="small",
                )

                x_off = 8
                y_off = 8

                # align legend labels to be above the handles
                for t, h in zip(leg.texts, leg.legend_handles):
                    t.set_ha("center")
                    t.set_position(
                        (t.get_position()[0] - x_off, t.get_position()[1] - y_off)
                    )


if __name__ == "__main__":
    exc_params = Params(
        gGABAs=[
            # 0,
            25,
            50,
            100,
            # 200,
        ],
        gAMPAs=np.round(np.arange(0, 20.0001, 5.0), 0),
        gNMDAs=[5.0, 7.5, 10.0],
        seeds=(None, 1013, 12987, 1234, 1837),
        __device_directory=f".cpp_{Params.fig_name}",
    )
    exc_params.run()
    exc_params.process()
    exc_params.plot(egabas=[-42, -56, -70])
    exc_params.save_figure(figs=exc_params.figs, close=True)

    plt.show()
