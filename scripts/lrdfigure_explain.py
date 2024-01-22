import time
from typing import Iterable
from matplotlib import patheffects
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
from brian2 import StateMonitor, mV, pA
from matplotlib.axes import Axes
from matplotlib.cbook import flatten
from matplotlib.pyplot import subplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import settings
from core.analysis import burst_stats, get_burst_periods
from core.lrdfigure import LRDFigure
from settings import text, logging, time_unit
from style.axes import adjust_spines, letter_axes
from style.plot_trace import (
    add_mg_ax,
    plot_conductance_zooms,
    plot_conductances,
    plot_diagram,
    plot_population_rates,
    plot_population_zooms,
    plot_state_average,
    plot_state_colorbar,
    plot_synaptic_var_zooms,
    plot_synaptic_variables,
)

logger = logging.getLogger(__name__)


class Explain(LRDFigure):
    """
    A class for generating and plotting LRDFigure 1, which shows the effect of varying the reversal potential of GABAergic
    synapses on network activity.

    Inherits from LRDFigure.

    Parameters
    ----------
    mv_step : int, optional
        Step in mV for the range of egaba values. Default is 2.
    time_per_value : int, optional
        Time in seconds for each egaba value. Default is 60.
    egaba : tuple of int, optional
        Range of egaba values (inclusive, exclusive). Default is (-78, -34).
    cache : bool, optional
        Whether to cache the simulation results. Default is False.

    Methods
    -------
    run(mv_step=2, time_per_value=60, egaba=(-78, -34), cache=False, **kwargs)
        Runs the simulation for the given parameters and returns the results.
    plot(timeit=True, plot_igaba=False, plot_rates=('r_all', 'r_I', 'r_E'), **kwargs)
        Plots the simulation results.

    """
    fig_name = "figure_1_explain"

    def run(
        self,
        mv_step=2,
        time_per_value=60,
        egaba: Iterable = (-78, -34),
        cache=False,  # still not plotting 100% when using cache
        **kwargs,
    ):
        """
        :param mv_step: step in mV
        :param time_per_value: time in seconds
        :param egaba: range of egaba values [inclusive, exclusive)
            useful to have the difference between the two values to be a multiple of the mv_step

        """
        assert egaba[0] < egaba[1], "egaba[0] must be smaller than egaba[1]"
        assert len(egaba) == 2, "egaba must be a list of two values"

        diff = egaba[1] - egaba[0]
        values = diff // mv_step
        steps = values - 1
        duration = values * time_per_value

        ehco3 = -18
        phco3 = 0.2
        pcl = 1 - phco3
        ecl = [round((e - phco3 * ehco3) / pcl, 2) for e in egaba]

        logger.info(f"EGABA range = {np.linspace(egaba[0], egaba[1], steps+2)[:-1]}")

        return super().run(
            duration=duration,
            num_ecl_steps=steps,
            E_Cl_0=ecl[0],
            E_Cl_end=ecl[-1],
            cache=cache,
            **kwargs,
        )

    def plot(
        self,
        timeit=True,
        plot_igaba=False,
        plot_rates=("r_all", "r_I", "r_E"),
        **kwargs,
    ):
        super().plot(**kwargs)
        logger.info(f"plotting {self.fig_name}")
        plot_time_start = time.time()
        gridspec = {
            "height_ratios": [
                1,  # Mg
                2,  # Egaba
                12,  # population rate
                6,  # conductances
                6,  # synaptic variable xs
                # 4,  # synaptic variable us
                # 2,  # synaptic variable w
            ]
        }

        ncols = 1
        fig, ax = subplots(
            nrows=gridspec["height_ratios"].__len__(),
            ncols=ncols,
            squeeze=False,
            sharex="col",
            gridspec_kw=gridspec,
            figsize=(settings.PAGE_H_FULL, settings.PAGE_W_FULL),
        )
        self.fig, self.ax = fig, ax
        fig.subplots_adjust(top=0.95, right=0.56, hspace=0.1)
        zoom_width = "70%"
        zoom_height = "95%"
        ax_col = ax[:, 0]
        ax_mg, ax_egaba, ax_pop, ax_con, ax_syn_x = ax_col
        ax_diagram = inset_axes(
            ax_pop,
            width=zoom_width,
            height="100%",
            loc="lower right",
            bbox_to_anchor=(0.8, 0.7, 1, 1),
            bbox_transform=ax_pop.transAxes,
        )
        add_mg_ax(self, ax_mg, time_unit=time_unit, append_axes=False)

        # letter_axes(ax_egaba, xy=(-0.25, 1.))
        letter_axes(list(flatten(ax[:2])), start="A", xy=(-0.25, 1.0))
        letter_axes(list(flatten(ax[2:])), start="D", xy=(-0.25, 1.0))
        letter_axes(ax_diagram, start="C", xy=(-0.1, 0.5))

        plot_diagram(self.N_E, self.N_I, ax_diagram)
        if isinstance(self.state_mon, pd.DataFrame):
            extent = [
                np.min(self.state_mon.I_GABA_rec.values),
                np.max(self.state_mon.I_GABA_rec.values),
            ]
        else:
            extent = [
                np.min(self.state_mon.I_GABA_rec) / pA,
                np.max(self.state_mon.I_GABA_rec) / pA,
            ]
        max_egaba = (
            np.max(self.state_mon.E_GABA.values)
            if isinstance(self.state_mon, pd.DataFrame)
            else np.max(self.state_mon.E_GABA)
        )
        min_egaba = (
            np.min(self.state_mon.E_GABA.values)
            if isinstance(self.state_mon, pd.DataFrame)
            else np.min(self.state_mon.E_GABA)
        )
        plot_state_colorbar(
            self.state_mon,
            "E_GABA",
            fig=fig,
            ax=ax_egaba,
            time_unit=time_unit,
            label_text=False,
            extent=extent if plot_igaba else None,
        )
        # cbar.set_ticks(np.linspace(min_egaba/mV, max_egaba/mV, self.num_ecl_steps+1))
        T = (
            self.state_mon.t[-1] / time_unit
            if isinstance(self.state_mon, StateMonitor)
            else self.state_mon.index[-1]
        )
        T = np.round(T)
        ax_egaba.tick_params(axis="x")

        if plot_igaba:
            plot_state_average(
                self.state_mon,
                ["I_GABA_rec"],
                var_unit=pA,
                var_names=["$I_{GABA}$"],
                ax=ax_egaba,
                alpha=0.8,
                only_mean=False,
                colors=["#90ee90"],
                lw=[0.1],
                time_unit=time_unit,
            )
            ax_egaba.locator_params(nbins=2, axis="y", tight=True)

        # rate zoom
        ax_insets = []
        num_xticks = 5
        xpad = 0.2

        to_plot = [getattr(self, rate) for rate in plot_rates]
        plot_population_rates(to_plot, ax=ax_pop, time_unit=time_unit, lw=0.1)

        burst_start_ts, burst_end_ts = burst_stats(
            self.r_all, time_unit=time_unit, plot_fig=False
        )
        logger.debug(f"burst_start_ts={burst_start_ts}")
        logger.debug(f"burst_end_ts={burst_end_ts}")
        periods = get_burst_periods(burst_start_ts, burst_end_ts)
        centered_periods = periods["mid"]
        if centered_periods.__len__() == 0:
            mid_xs = None
            xlim_samples = []
            burst_samples = []
        else:
            num_bursts = len(centered_periods)
            idxs = [num_bursts // 2]
            xlim_samples = []
            burst_samples = []
            mid_xs = []
            for idx in idxs:
                xlim_sample = centered_periods[idx]
                trim_x = (xlim_sample[1] - xlim_sample[0]) / 3
                xlim_sample = (xlim_sample[0] + trim_x, xlim_sample[1] - trim_x)
                mid_xs.append((xlim_sample[1] + xlim_sample[0]) / 2)
                xlim_samples.append(xlim_sample)
                burst_samples.append(
                    (*periods["burst"][idx], np.diff(periods["interburst"][idx])[0])
                )

        ax_insets += plot_population_zooms(
            ax_pop,
            xlim_samples,
            width=zoom_width,
            height=zoom_height,
            xticks=num_xticks,
            xpad=xpad,
            yticks=8,
            time_unit=time_unit,
            rel_lw=1.0,
            loc1=2,
            loc2=3,
        )
        mid_xs = []
        for ax_inset_pop in ax_insets:
            xdata, ydata = ax_inset_pop.get_lines()[0].get_xydata().T
            xdata_E, ydata_E = ax_inset_pop.get_lines()[1].get_xydata().T
            max_idx = np.argmax(ydata)
            mid_xs.append(xdata[max_idx])
            for mid_x, xlim_sample, burst_sample in zip(
                mid_xs, xlim_samples, burst_samples
            ):
                # center of burst
                ax_inset_pop.axvline(mid_x, lw=1, c=settings.COLOR.K, alpha=0.5)
                x1, x2, interburst = burst_sample
                burst_mask = (xdata >= x1) & (xdata <= x2)
                burst_x = xdata[burst_mask]
                burst_y = ydata[burst_mask] * 0.9
                ax_inset_pop.axvline(mid_x, lw=1, c=settings.COLOR.K, alpha=0.5)

                # annotate amplitude
                rate_baseline_I = np.mean(ydata[xdata <= x1])
                rate_baseline_E = np.mean(ydata_E[xdata_E <= x1])
                rate_baseline = (
                    rate_baseline_I * self.N_I + rate_baseline_E * self.N_E
                ) / self.N
                rate_max_I = np.max(ydata)
                rate_max_E = np.max(ydata_E)
                rate_max_mean = (rate_max_I * self.N_I + rate_max_E * self.N_E) / self.N
                ax_inset_pop.annotate(
                    f"$\\approx ${rate_max_mean - rate_baseline:.2f} Hz",
                    xy=(xlim_sample[-1] - 1, rate_baseline),
                    xytext=(xlim_sample[-1] - 1, rate_max_mean),
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                    c=settings.COLOR.K,
                    alpha=0.8,
                    arrowprops=dict(
                        arrowstyle="<|-|>", color=settings.COLOR.K, alpha=0.8
                    ),
                )
                ax_inset_pop.annotate(
                    f"$\\approx ${rate_max_I - rate_baseline_I:.2f} Hz",
                    xy=(xlim_sample[-1] - 0.9, rate_baseline_I),
                    xytext=(xlim_sample[-1] - 0.9, rate_max_I),
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                    c=settings.COLOR.inh,
                    alpha=0.8,
                    arrowprops=dict(
                        arrowstyle="<|-|>",
                        color=settings.COLOR.inh,
                        alpha=0.5,
                        lw=0.5,
                    ),
                )
                ax_inset_pop.annotate(
                    f"$\\approx ${rate_max_E - rate_baseline_E:.2f} Hz",
                    xy=(xlim_sample[-1] - 0.8, rate_baseline_E),
                    xytext=(xlim_sample[-1] - 0.8, rate_max_E),
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                    c=settings.COLOR.R,
                    alpha=0.8,
                    arrowprops=dict(
                        arrowstyle="<|-|>", color=settings.COLOR.R, alpha=0.5, lw=0.5
                    ),
                )
                # annotate interburst
                ax_inset_pop.annotate(
                    f"$\\approx ${interburst:.2f} s",
                    xy=(xlim_sample[0] / 2 + burst_x[0] / 2, burst_y[0]),
                    ha="center",
                    va="bottom",
                    fontsize="xx-small",
                    c=settings.COLOR.K,
                    alpha=0.8,
                )
                ax_inset_pop.annotate(
                    "",
                    xy=(xlim_sample[0], burst_y[0]),
                    xytext=(burst_x[0], burst_y[0]),
                    arrowprops=dict(
                        arrowstyle="<|-", lw=1, ls="--", color=settings.COLOR.K, alpha=1
                    ),
                )
                # annotate burst width
                ax_inset_pop.annotate(
                    f"{burst_x[-1] - burst_x[0]:.2f} s",
                    xy=(burst_x[-1] / 2 + burst_x[0] / 2, burst_y[0]),
                    ha="center",
                    va="bottom",
                    fontsize="xx-small",
                    c="k",
                    alpha=0.8,
                    bbox=dict(boxstyle="Square, pad=0.2", color="w", alpha=0.5),
                )
                ax_inset_pop.annotate(
                    "",
                    xy=(burst_x[0], burst_y[0]),
                    xytext=(burst_x[-1], burst_y[0]),
                    arrowprops=dict(arrowstyle="<|-|>", lw=1, color="k", alpha=1),
                )
                ax_inset_pop.yaxis.set_major_locator(MaxNLocator(5))

        only_mean = True
        plot_conductances(
            self.state_mon,
            nrn_idx_type=self.nrn_idx_type,
            ax=ax_con,
            only_mean=only_mean,
            time_unit=time_unit,
            lw=0.1,
        )
        ax_con.get_legend().remove()
        ax_insets += plot_conductance_zooms(
            ax_con,
            xlim_samples,
            conductances=True,
            connections=False,
            width=zoom_width,
            height=zoom_height,
            xticks=num_xticks,
            xpad=xpad,
            show_legend=False,
            time_unit=time_unit,
            lw=0.5,
        )
        syn_perc = True
        subsample = 250  # faster and smoother plotting
        plot_synaptic_variables(
            self.synapse_monitors,
            self.sp_all,
            self.tau_d,
            self.tau_f,
            perc=syn_perc,
            nrn_idx=self.nrn_idx,
            ax_x=ax_syn_x,
            ax_u=None,
            ax_w=None,
            time_unit=time_unit,
            subsample=subsample,
            lw=0.1,
        )
        w_max = np.max([syn_mon.dI_S.max() for syn_mon in self.synapse_monitors])
        w_max = w_max * 100 if syn_perc else w_max
        ax_syn_insets = plot_synaptic_var_zooms(
            ax_syn_x,
            xlims=xlim_samples,
            width=zoom_width,
            height=zoom_height,
            xticks=num_xticks,
            xpad=xpad,
            yticks=5,
            time_unit=time_unit / subsample,
            lw=1.0,
            loc1=2,
            loc2=3,
        )
        ax_insets += ax_syn_insets
        for ax_inset in ax_syn_insets:
            for line in ax_inset.get_lines():
                x_data, y_data = line.get_xydata().T
                max_y_idx = np.argmax(y_data)
                max_y = y_data[max_y_idx]
                max_x = x_data[max_y_idx]
                ax_inset.annotate(
                    "",
                    xy=(max_x, max_y * 0.9),
                    xytext=(0, -5),
                    textcoords="offset points",
                    va="center",
                    ha="center",
                    c=line.get_color(),
                    arrowprops=dict(
                        arrowstyle="wedge",
                        facecolor=line.get_color(),
                        ec=line.get_color(),
                    ),
                )

        for ax_inset in ax_insets:
            ax_inset.yaxis.set_label_position("right")
            for mid_x, xlim_sample in zip(mid_xs, xlim_samples):
                if ax_inset.get_xlim()[0] == xlim_sample[0]:
                    ax_inset.axvline(mid_x, lw=1, c=settings.COLOR.K, alpha=0.5)

        if len(ax_insets) > 1:
            _num_samples = len(xlim_samples)
            for ax_inset in ax_insets[_num_samples:-_num_samples:_num_samples]:
                ax_inset.set_xticklabels([])

        num_ecl_vals = self.num_ecl_steps + 1  # +1 for number of values (steps+1)
        t_steps = np.linspace(0, T, num_ecl_vals + 1)  # +1 for first tick
        t_steps_off = t_steps[1] / 2
        e_ticks = (t_steps + t_steps_off)[:-1]  # don't need last tick
        e_ticklabels = np.linspace(min_egaba / mV, max_egaba / mV, num_ecl_vals)

        if np.all(e_ticklabels == np.round(e_ticklabels).astype(int)):
            e_ticklabels = e_ticklabels.astype(int)
        else:
            e_ticklabels = np.round(e_ticklabels, 2)

        for i, (x, e) in enumerate(zip(e_ticks, e_ticklabels)):
            ax_egaba.annotate(
                f"{e}",
                xy=(x, 0.5),
                fontsize="x-small",
                ha="center",
                va="center",
                rotation=0,
                c="w",
                # path_effects=[
                #     patheffects.withStroke(linewidth=0.1, foreground="black")
                # ],
            )
        ax_egaba.set_ylabel(f"{text.EGABA}\n(mV)")
        ax_egaba.tick_params(
            axis="x", which="minor", top=False, bottom=False, labeltop=True
        )
        ax_egaba.tick_params(
            axis="x", which="major", top=False, bottom=False, labeltop=False
        )

        ax[-1, 0].set_xticks(t_steps[::2])
        for _ax in ax[2:, 0]:
            _ax.set_xticks(t_steps, minor=True)
            _ax.grid(True, axis="x", which="minor", ls="--", lw=0.5, alpha=0.4)
            _ax.grid(True, axis="x", which="major", ls="--", lw=0.5, alpha=0.4)
        adjust_spines([ax_diagram], [], position=0)
        ax_egaba.spines["top"].set_visible(False)
        adjust_spines(ax, ["left"], sharedx=True, position=0)
        adjust_spines(ax_egaba, [], position=0, sharedy=True)
        adjust_spines(ax_mg, [], position=0, sharedy=True)
        ax[-1, 0].set_xbound(0, T)
        ax_egaba.set_xbound(0, T)

        # ax_igaba.set_ylabel(ax_igaba.get_ylabel(), rotation=0, va='center_baseline', ha='left')

        # adjust_ylabels(ax_col)

        for ax_i in ax[:-1, 0]:
            ax_i.xaxis.label.set_visible(False)
        ax[-1, 0].set_xlabel(f"{text.TIME}" + " (%s)" % time_unit)
        fig.align_ylabels(list(flatten(ax)))
        plot_time = time.time()
        plot_dt = plot_time - plot_time_start
        if timeit:
            logger.info("took {:.2f}s to plot".format(plot_dt))
        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(1337)
    explain = Explain()

    explain.run(
        mv_step=2,
        time_per_value=60,
        egaba=[-78, -34],
    )

    explain.plot()
    explain.save_figure()
    plt.show()
