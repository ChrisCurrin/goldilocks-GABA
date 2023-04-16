import itertools
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from core.analysis import burst_stats
from core.lrdfigure import MultiRunFigure
from settings import constants, logging, time_unit

logger = logging.getLogger(__name__)

round_EGABA = f"{constants.E_GABA}"


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
        time_per_value=60,
        EGABA_0=-74,
        EGABA_end=-40,
        mv_step=2,
        **kwargs,
    ):
        self.gGABAs = gGABAs
        self.gAMPAs = gAMPAs
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
        duration = values * time_per_value
        ecl_0 = round((EGABA_0 - phco3 * ehco3) / pcl, 2)
        ecl_end = round((EGABA_end - phco3 * ehco3) / pcl, 2)

        kwargs["E_Cl_0"] = ecl_0
        kwargs["E_Cl_end"] = ecl_end

        manual_cl = kwargs.pop("manual_cl", True)

        super().__init__(
            OrderedDict(
                g_GABA_max={"range": self.gGABAs, "title": constants.G_GABA},
                g_AMPA_max={"range": self.gAMPAs, "title": constants.G_AMPA},
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
                constants.EGABA,
                constants.G_GABA,
                constants.G_AMPA,
                "Burst start time (s)",
            ]
        )

        for gGABA, gAMPA, run_idx in itertools.product(
            self.gGABAs, self.gAMPAs, run_idxs
        ):
            df_egaba = self.df[gGABA, gAMPA, run_idx, "E_GABA_all"]
            df_rates = self.df[gGABA, gAMPA, run_idx, "r_all"]
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
                        start_t,
                    ]
                if len(burst_ts) == 0:
                    # add expected observations
                    df_g_E.loc[df_g_E.shape[0]] = [run_idx, egaba, gGABA, gAMPA, np.nan]

        # analysis
        df_g_E["bin"] = pd.cut(
            df_g_E["Burst start time (s)"],
            bins=np.append(bins, bins[-1] + bin_size),
            labels=bins.astype(int),
        )

        num_bursts_col = f"Number of bursts\n(per {bin_size} s)"
        df_g_E_bursts = (
            df_g_E.groupby(
                [constants.EGABA, constants.G_GABA, constants.G_AMPA, "bin", "run_idx"]
            )
            .size()
            .to_frame()
            .reset_index()
            .rename(columns={0: num_bursts_col})
        )
        df_g_E_bursts["hyperpolaring\nEGABA"] = pd.cut(
            df_g_E_bursts[constants.EGABA],
            bins=[-100, self.vcenter, 0],
            labels=[True, False],
        )

        self.df_g_E = df_g_E
        self.df_g_E_bursts = df_g_E_bursts
        self.num_bursts_col = num_bursts_col

    def plot(self, timeit=True, **kwargs):
        super().plot(**kwargs)
        logger.info("plotting")
        plot_time_start = time.time()

        self.plot_gampa()

        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self

    def plot_gampa(self, ax=None, single_plot=False):
        if ax is None:
            fig, ax = plt.subplots(1 + int(single_plot), sharex=True, sharey=True)
        df_g_E_bursts = self.df_g_E_bursts
        num_bursts_col = self.num_bursts_col

        # bursts v G with G_AMPA
        if single_plot:
            sns.barplot(
                x=constants.G_GABA,
                y=num_bursts_col,
                hue=constants.G_AMPA,
                hue_order=sorted(
                    df_g_E_bursts[constants.G_AMPA].unique(), reverse=True
                ),
                # err_style='band', err_kws=dict(alpha=0.01, ),
                errwidth=0.5,
                orient="v",
                palette="YlOrRd_r",
                data=df_g_E_bursts[df_g_E_bursts[constants.EGABA] >= -55],
                ax=ax,
            )
            leg = ax.legend(
                ncol=len(df_g_E_bursts[constants.G_AMPA].unique()),
                loc=(0, 0.9),
                handlelength=0.5,
                handletextpad=0.0,
                columnspacing=0.5,
                mode="expand",
                fontsize="xx-small",
                frameon=False,
                title=f"{constants.G_AMPA} (ns)",
                title_fontsize="small",
            )
            ax.add_artist(leg)
            sns.barplot(
                x=constants.G_GABA,
                y=num_bursts_col,
                hue=constants.G_AMPA,
                hue_order=sorted(
                    df_g_E_bursts[constants.G_AMPA].unique(), reverse=True
                ),
                # err_style='band', err_kws=dict(alpha=0.01, ),
                errwidth=0.5,
                orient="v",
                palette="YlGnBu_r",
                data=df_g_E_bursts[df_g_E_bursts[constants.EGABA] <= -55],
                ax=ax,
            )
            lines = [
                Line2D([], [], c=sns.color_palette("Reds", 1)[0]),
                Line2D([], [], c=sns.color_palette("Blues", 1)[0]),
            ]
            labels = ["depolarising", "hyperpolarising"]
            ax.legend(
                lines,
                labels,
                ncol=1,
                loc=(1, 0),
                fontsize="x-small",
                frameon=False,
                title=f"{constants.EGABA}",
                title_fontsize="small",
            )
        else:
            sns.barplot(
                x=constants.G_AMPA,
                y=num_bursts_col,
                hue=constants.G_GABA,
                hue_order=sorted(
                    df_g_E_bursts[constants.G_GABA].unique(), reverse=False
                ),
                # err_style='band', err_kws=dict(alpha=0.01, ),
                errwidth=0.5,
                orient="v",
                palette="Greens",
                alpha=0.5,
                data=df_g_E_bursts[
                    (df_g_E_bursts[constants.EGABA] == -56)
                    & (df_g_E_bursts[constants.G_GABA] >= 0)
                ],
                ax=ax[0],
            )
            leg = ax[0].legend(
                ncol=len(df_g_E_bursts[constants.G_GABA].unique()),
                loc=(0, 0.9),
                handlelength=0.5,
                handletextpad=0.0,
                columnspacing=0.5,
                fontsize="xx-small",
                frameon=False,
                title=f"{constants.G_GABA} (ns)",
                title_fontsize="small",
            )
            sns.barplot(
                x=constants.G_AMPA,
                y=num_bursts_col,
                hue=constants.G_GABA,
                # hue_order=sorted(df_g_E_bursts[constants.G_GABA].unique(), reverse=True),
                # err_style='band', err_kws=dict(alpha=0.01, ),
                errwidth=0.5,
                orient="v",
                palette="Greens",
                zorder=-99,
                alpha=0.5,
                data=df_g_E_bursts[
                    (df_g_E_bursts[constants.EGABA] == -48)
                    & (df_g_E_bursts[constants.G_GABA] >= 0)
                ],
                ax=ax[1],
            )
            ax[1].legend().remove()


if __name__ == "__main__":
    gve = Params(
        gGABAs=np.append(np.round(np.arange(0, 100.0001, 10), 0), [125, 160, 200, 250]),
        gAMPAs=np.round(np.arange(0, 10.0001, 1), 0),
    )
    gve.run()
    gve.process()
    gve.plot()
    gve.save_figure(figs=gve.figs, close=True)

    plt.show()
