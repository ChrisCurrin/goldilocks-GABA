import logging
from typing import Union

import brian2.numpy_ as np
import pandas as pd
import spectra
from brian2 import defaultclock
from brian2.monitors import PopulationRateMonitor
from brian2.units import Hz, ms, second, uS, nS, Quantity
from brian2tools import plot_rate, plot_raster, plot_state, plot_synapses
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

import settings
from settings import COLOR, text
from style.axes import adjust_spines, colorbar_inset, create_zoom

logger = logging.getLogger("brian2viz")

default_colorbar_props = dict(fraction=0.1, pad=0.05, aspect=10)
SMOOTH_RATE_ARGS = dict(window="flat", width=10.1 * ms)


def perc2frac(perc: str):
    """Convert a string percentage to a fraction"""
    return float(perc[:-1]) / 100


def frac2perc(frac: float):
    """Cnvert a numeric fraction to a string percetange"""
    return f"{100*frac}%"


def spike_train_profile(spike_mon, time_unit=ms, ax=None):
    """

    :param spike_mon:
    :type spike_mon: : SpikeMonitor
    :param time_unit:
    :type time_unit:
    :param ax:
    :type ax: ndarray[Axes] or Tuple[Figure,ndarray[Axes]]

    """
    import pyspike as spk

    def spike_train_from_spike_monitor(sp_mon):
        """
        Convert spike trains from Brian2 simulator's SpikeMonitor class to PySpike's SpikeTrain class
        Note that time is converted to seconds so if a simulation was 1000 ms, the edge would be (0.,1.)
        :param sp_mon: SpikeMonitor object to retrieve spike times from
        :type sp_mon: :class: brian2.SpikeMonitor
        @return: list of SpikeTrain objects for analysis using PySpike methods (e.g. isi_profile, spike_sync_profile)
        """
        brian2_spk_trains = sp_mon.spike_trains()
        edges = (0, sp_mon.t[-1] / time_unit)
        _spike_trains = []
        for idx in brian2_spk_trains:
            _spike_trains.append(
                spk.SpikeTrain(brian2_spk_trains[idx] / time_unit, edges=edges)
            )
        return _spike_trains

    ##
    # Spike train analysis
    ##
    logger.info("spike train profile")
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=1)
    elif type(ax) is tuple:
        fig, ax = ax

    spike_trains = spike_train_from_spike_monitor(spike_mon)

    avrg_spike_sync_profile = spk.spike_sync_profile(spike_trains)

    kwargs = dict(
        linestyle="-", color=COLOR.K, alpha=0.5, lw=0.1, rasterized=settings.RASTERIZED
    )

    x, y = avrg_spike_sync_profile.get_plottable_data()
    ax[-1].plot(x, y, **kwargs)
    ax[-1].set(ylim=(0, 1), ylabel="SPIKE-Sync\n(AU)")
    logger.info("SPIKE synchronisation: %.8f" % avrg_spike_sync_profile.avrg())
    adjust_spines(ax[-1], ["left"], sharedx=True)


def plot_rates_from_spikes(
    spike_monitor, N=0, ax=None, bin_size=1 * ms, time_unit=ms, **kwargs
):
    if type(spike_monitor) is list:
        rec_vars = dict(vars())
        del rec_vars["kwargs"]
        del rec_vars["spike_monitor"]
        for spk_mon in spike_monitor:
            plot_rates_from_spikes(spk_mon, **rec_vars, **kwargs)
        return
    logger.debug("Rates from spike times")
    # Generate frequencies
    i, t = spike_monitor.it
    duration = t[-1]
    if N == 0:
        N = max(i) + 1
    spk_count, bin_edges = np.histogram(t / time_unit, int(duration / ms))
    rate = np.double(spk_count) / N / bin_size / Hz

    pre_ax = ax
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(bin_edges[:-1], rate, **kwargs)

    if pre_ax is None:
        ax.set(xlim=(0.0, duration / ms), xlabel="time (ms)", ylabel="rate (Hz)")


def plot_population_rates(
    rate_monitors, smooth=True, time_unit=ms, ax=None, leg_lw=4.0, **kwargs
):
    """Plot each rate monitor with some alpha and according to color scheme in settings.COLOR

    :param rate_monitors: List of rate monitors
    :type rate_monitors: list[PopulationRateMonitor] or list[pd.DataFrame]
    :param smooth:
    :param time_unit:
    :param ax:
    :type ax: Axes
    """
    logger.info("Population rates")
    if ax is None:
        fig, ax = plt.subplots()

    smooth_rate_args = process_smooth_args(smooth)

    if "lw" in kwargs:
        lw = kwargs["lw"]
        del kwargs["lw"]
    elif "linewidth" in kwargs:
        lw = kwargs["linewidth"]
        del kwargs["linewidth"]
    else:
        lw = 1.0

    colors = kwargs.pop("colors", None)
    if colors is None:
        colors = [kwargs["color"]] * len(rate_monitors) if "color" in kwargs else None

    leg_handles_labels = []
    for r, rate_mon in enumerate(rate_monitors):
        if colors:
            color = colors[r]
        else:
            color = (
                COLOR.RATE_dict[rate_mon.name]
                if rate_mon.name in COLOR.RATE_dict
                else None
            )
        _rate = rate_mon.smooth_rate(**smooth_rate_args) if smooth else rate_mon.rate

        if rate_mon.name in text.POPULATION_RATE_MAP:
            label = text.POPULATION_RATE_MAP[rate_mon.name]
        else:
            label = rate_mon.name
        plot_rate(
            rate_mon.t,
            _rate,
            time_unit=time_unit,
            axes=ax,
            color=color,
            alpha=0.6,
            lw=lw,
            label=label,
            **kwargs,
        )
        leg_handles_labels.append(
            (
                plt.Line2D(
                    [],
                    [],
                    lw=lw * leg_lw,
                    color=color,
                    alpha=0.6,
                    label=label,
                    **kwargs,
                ),
                label,
            )
        )
    ax.legend(*list(zip(*leg_handles_labels)), frameon=False)


def process_smooth_args(smooth: Union[bool, dict]):
    if type(smooth) is dict:
        smooth_rate_args = smooth
    elif smooth:
        smooth_rate_args = SMOOTH_RATE_ARGS
    else:
        smooth_rate_args = dict()
    return smooth_rate_args


def plot_population_zooms(
    ax_to_zoom, xlims, width="100%", height="100%", xpad=0.1, time_unit=ms, **kwargs
):
    logger.debug("creating population zooms")
    ax_zooms = []

    if perc2frac(width) > 0.5 and xlims.__len__() > 1:
        width = frac2perc(perc2frac(width) / xlims.__len__())

    for x_i, xlim_sample in enumerate(xlims):
        bbox_start = 1.0 + x_i * (perc2frac(width) + x_i * xpad)
        ax_inset = create_zoom(
            ax_to_zoom,
            (width, height),
            loc="center left",
            xlim=xlim_sample,  # before last oscillation
            ylim=0,
            xunit=defaultclock.dt / time_unit,
            inset_kwargs=dict(
                bbox_to_anchor=(bbox_start, 0, 1, 1),
                bbox_transform=ax_to_zoom.transAxes,
            ),
            **kwargs,
        )
        ax_inset.yaxis.tick_right()
        ax_inset.yaxis.set_label_position("right")
        ax_inset.set_ylabel("(Hz)", color=COLOR.K)
        # ax_inset.tick_params(which='major', axis='y', labelright=True)
        ax_zooms.append(ax_inset)
    return ax_zooms


def plot_spectrogram(arr, time_unit=ms, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    NFFT = np.power(2, 10)  # the length of the windowing segments
    Fs = int(time_unit / second / defaultclock.dt)  # the sampling frequency
    logger.info("Spectrogram rates")
    power_spectrum, freqs, t_bins, im = ax.specgram(
        arr,
        NFFT=NFFT,
        Fs=Fs,
        mode="psd",
        scale="default",
        # cmap='Spectral_r',
        # norm=matplotlib.colors.Normalize(-30, 30),
        rasterized=settings.RASTERIZED,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.1, pad=0.0)
    cbar.set_label("Power (dB/Hz)")
    ax.set(ylabel="Frequency (Hz)")
    # ax.set_ylim(0, 2)


def plot_conductances(
    state_mon,
    gs=None,
    nrn_idx_type=None,
    only_mean=False,
    ax=None,
    time_unit=ms,
    **kwargs,
):
    if gs is None:
        gs = ["AMPA", "GABA", "NMDA"]
    g_var_names = []
    g_legend_names = []
    g_colors = []
    for g in gs:
        if "g_" in g:
            g = g.replace("g_", "")
        g_var_names.append(f"g_{g}")
        g_legend_names.append(f"$g_{{{g}}}$")
        g_colors.append(COLOR.g_dict[g])

    logger.info(f"Plot postsynaptic conductances - {g_var_names}")

    plot_state_average(
        state_mon,
        g_var_names,
        var_unit=nS,
        ax=ax,
        alpha=0.8,
        colors=g_colors,
        blend=nrn_idx_type,
        linestyles="-",
        only_mean=only_mean,
        window=10.0 * ms,
        time_unit=time_unit,
        **kwargs,
    )

    ax.axhline(color=COLOR.K, alpha=0.2, lw=0.5, linestyle=":")

    ax.legend(
        g_legend_names
        # , bbox_to_anchor=(1, 0), loc=3
    )
    ax.set(ylabel="conductance (nS)")

    sum_g_NMDA = np.sum(state_mon.g_NMDA, axis=1)
    sum_g_AMPA = np.sum(state_mon.g_AMPA, axis=1)
    sum_g_GABA = np.sum(state_mon.g_GABA, axis=1)

    net_g = sum_g_NMDA + sum_g_AMPA - sum_g_GABA
    g_vars = {
        "$g_{NMDA}$": sum_g_NMDA,
        "$g_{AMPA}$": sum_g_AMPA,
        "$g_{GABA}$": sum_g_GABA,
        r"$\sum $": net_g,
    }
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        for g_var_k, g_var_v in g_vars.items():
            _g_str = ", ".join(["{:>6.2f}".format(g_var / uS) for g_var in g_var_v])
            logger.debug(f"\t{g_var_k} = {_g_str} (uS)")
    logger.info(f"\t plotted - Net g = {np.mean(net_g / uS):>6.2f} (uS)")


def plot_conductance_zooms(
    ax,
    xlims,
    conductances=None,
    connections=None,
    width="100%",
    height="100%",
    xticks=True,
    xpad=0.1,
    ypad=0.04,
    show_legend=False,
    time_unit=ms,
    **kwargs,
):
    logger.info("Plot conductance zooms")
    conductances = _process_conductances_arg(conductances)
    connections = _process_connections_arg(connections)

    ax_insets = []

    # calculate width and height per box
    if perc2frac(width) > 0.5 and xlims.__len__() > 1:
        width = frac2perc(perc2frac(width) / xlims.__len__())

    g_height, g_bbox_y = _calc_zoom_params(conductances, height, ypad, show_legend)

    c_height, c_bbox_y = _calc_zoom_params(connections, height, ypad, show_legend)

    # get some lines
    lines = ax.get_lines()
    only_mean = lines.__len__() <= 4
    l_i_0 = 3  # line index start (after mean traces)
    num_traces = (lines.__len__() - 1) // 3 - 1  # -1 (zero) / 3 (each g) - 1 (mean)
    handles, labels = ax.get_legend_handles_labels()
    for x_i, xlim_sample in enumerate(xlims):
        if xlim_sample is None:
            continue
        logger.debug(f"\t xlim = {xlim_sample}")
        bbox_start = 1.0 + x_i * (perc2frac(width) + x_i * xpad)
        logger.debug("\t conductance-specific")
        for i, label in enumerate(conductances):
            logger.debug(f"\t\t {label}")
            if label == "all":
                included_lines = None
            else:
                l_i = i * num_traces + l_i_0
                included_lines = (
                    list(np.array(lines)[np.arange(l_i, l_i + num_traces, 1)])
                    if not only_mean
                    else []
                )
                included_lines += [lines[i], lines[-1]]  # avg and 0 lines

            ax_inset = create_zoom(
                ax,
                (width, g_height),
                lines=included_lines,
                loc="lower left",
                xlim=xlim_sample,
                xticks=False,
                yticks=2,
                ec=COLOR.g_dict[label],
                xunit=defaultclock.dt / time_unit,
                inset_kwargs=dict(
                    bbox_to_anchor=(bbox_start, i * g_bbox_y + i * ypad, 1, 1),
                    bbox_transform=ax.transAxes,
                ),
                box_kwargs=dict(linestyle="--", zorder=99 + x_i),
                connector_kwargs=dict(color="None"),  # remove connectors
                **kwargs,
            )
            ax_inset.yaxis.tick_right()
            ax_inset.yaxis.set_label_position("right")
            # ax_inset.set_ylabel(f"$g_{{{label}}}$", rotation=0, ha='left')
            ax_inset.set_ylabel("(nS)", color=COLOR.g_dict[label])
            # ax_inset.legend([lines[i]], [f"$g_{{{label}}}$"], loc='upper right', frameon=False)
            ax_inset.annotate(
                f"$g_{{{label}}}$",
                xy=(0.9, 0.9),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize="small",
                color=COLOR.g_dict[label],
            )

            if show_legend and i == conductances.__len__() - 1:
                leg_labels = ["→ E", "→ I"]
                leg_lines = []
                for leg_label in leg_labels:
                    leg_lines.append(
                        plt.Line2D([], [], ls="-", c=COLOR.CONN_BLEND[leg_label[-1]])
                    )
                ax_inset.legend(
                    leg_lines,
                    leg_labels,
                    loc=3,
                    handlelength=1,
                    bbox_to_anchor=(0, 1, 1, 1),
                    ncol=leg_labels.__len__(),
                    mode="expand",
                    fontsize="x-small",
                )
            elif i == 0 and xticks:
                ax_inset.xaxis.set_visible(True)
                nbins = xticks if type(xticks) is int else 3
                ax_inset.locator_params(nbins=nbins, axis="x", tight=True)
            ax_insets.append(ax_inset)
        logger.debug("\t plot connection-specific zooms")
        for i, label in enumerate(connections):
            logger.debug(f"\t\t {label}")
            if label == "all":
                included_lines = None
            else:
                if label.endswith("_E"):
                    l_start = i + l_i_0
                    l_end = i + num_traces * 2 + 2
                    l_step = 2
                else:
                    l_start = int(l_i_0 + num_traces * 2 + i % 2 * num_traces / 2)
                    l_end = int(l_i_0 + num_traces * 2 + (i + 1) % 2 * num_traces / 2)
                    l_step = 1
                avg_lines = lines[0:2] if label.endswith("_E") else lines[2:3]
                included_lines = (
                    list(np.array(lines)[np.arange(l_start, l_end, l_step)])
                    if not only_mean
                    else []
                )
                included_lines += avg_lines + [lines[-1]]  # avg and 0 lines
            ax_inset = create_zoom(
                ax,
                (width, c_height),
                lines=included_lines,
                loc="lower left",
                xlim=xlim_sample,
                xticks=False,
                yticks=2,
                ec=COLOR.CONN_dict[label],
                xunit=defaultclock.dt / time_unit,
                inset_kwargs=dict(
                    bbox_to_anchor=(bbox_start, c_bbox_y * i, 1, 1),
                    bbox_transform=ax.transAxes,
                ),
                connector_kwargs=dict(color="None"),
                **kwargs,
            )
            ax_inset.yaxis.tick_right()
            ax_inset.yaxis.set_label_position("right")
            # ax_inset.set_ylabel(settings.CONNECTION_MAP[label], rotation=0)
            ax_inset.legend(lines[i], settings.CONNECTION_MAP[label], frameon=False)

            if show_legend and i == conductances.__len__() - 1:
                leg_labels = [
                    "$g_{}$".format(g) for g in ["{NMDA}", "{AMPA}", "{GABA}"]
                ]
                leg_lines = []
                for leg_label, ls in zip(leg_labels, ["--", ":", "-"]):
                    leg_lines.append(
                        plt.Line2D([], [], ls=ls, c=COLOR.g_dict[leg_label[-6:-2]])
                    )
                ax_inset.legend(
                    leg_lines,
                    leg_labels,
                    loc=3,
                    handlelength=1,
                    bbox_to_anchor=(0, 1, 1, 1),
                    ncol=len(leg_labels),
                    fontsize="x-small",
                    mode="expand",
                )
            elif i == 0:
                ax_inset.xaxis.set_visible(True)
                ax_inset.locator_params(nbins=3, axis="x", tight=True)
            ax_insets.append(ax_inset)
    return ax_insets


def _calc_zoom_params(collection, height, ypad, show_legend):
    if len(collection) >= 1:
        _height_f = perc2frac(height) - 0.1 if show_legend else perc2frac(height)
        height = frac2perc(_height_f / len(collection))
        bbox_y = (_height_f - ypad) / len(collection)
    else:
        height = 0
        bbox_y = 0
    return height, bbox_y


def _process_connections_arg(connections):
    if connections is None or connections is True:
        connections = ["C_I_I", "C_E_I", "C_I_E", "C_E_E"]  # bottom to top
    elif connections is False:
        connections = []
    return connections


def _process_conductances_arg(conductances):
    if conductances is None or conductances is True:
        conductances = [
            "AMPA",
            "GABA",
            "NMDA",
        ]  # bottom to top (should be same order as `plot_conductances`
    elif conductances is False:
        conductances = []
    return conductances


def plot_synaptic_variables(
    synapse_monitors,
    spike_monitor,
    tau_d=0,
    tau_f=0,
    perc=True,
    nrn_idx=None,
    marker=False,
    ax_x=None,
    ax_u=None,
    ax_w=None,
    time_unit=ms,
    subsample=1000,
    **kwargs,
):
    """Synaptic variables
    Retrieves indexes of spikes in the synaptic monitor using the fact that we are sampling spikes and synaptic
    variables by the same dt.
    A lot of computation is done reconstructing spike times so subsampling by an order of magnitude dramatically
    decreases the time taken for this method.
    """
    logger.info("Synaptic variables")
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.8
    if "ms" not in kwargs and "markersize" not in kwargs:
        kwargs["markersize"] = 1
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 0.5
    mul = 100 if perc else 1
    ss = subsample
    for s_i, synapse_mon in enumerate(synapse_monitors):
        s_name = (
            synapse_mon.name
            if isinstance(synapse_mon, pd.DataFrame)
            else synapse_mon.source.name
        )
        t = synapse_mon.t[0] if isinstance(synapse_mon, pd.DataFrame) else synapse_mon.t
        t = t[::ss]
        logger.debug("\t {}".format(s_name))
        if nrn_idx.__len__() == 2:
            s_i = 0  # only a single recording is done per monitor
        elif s_i > nrn_idx.__len__():
            s_i %= 2  # 2 recordings are done per monitor

        spike_times = spike_monitor.t[
            spike_monitor.i == nrn_idx[s_i]
        ]  # where index is target
        var_idx = s_i % 2
        logger.debug("\t spike_times calculated")
        spk_index = np.in1d(t, spike_times)
        logger.debug("\t spk_index calculated")

        if marker:
            if tau_d > 0 and ax_x is not None:
                ax_x.plot(
                    t[spk_index] / time_unit,
                    mul * synapse_mon.x_S[var_idx][::ss][spk_index],
                    "x",
                    color=COLOR.CONN_dict[s_name],
                    label=None,
                    rasterized=settings.RASTERIZED,
                    **kwargs,
                )
                logger.debug("\t plotted ax_x points")
            if tau_f > 0 and ax_u is not None:
                ax_u.plot(
                    t[spk_index] / time_unit,
                    mul * synapse_mon.u_S[var_idx][::ss][spk_index],
                    ".",
                    color=COLOR.CONN_dict[s_name],
                    label=None,
                    rasterized=settings.RASTERIZED,
                    **kwargs,
                )

        # Super-impose reconstructed solutions
        t_spk = (
            np.copy(t) if isinstance(t, pd.DataFrame) else Quantity(t, copy=True)
        )  # Continuous spike times
        logger.debug("\t copied t")
        # Continuous spike times
        for ts in spike_times:
            t_spk[t >= ts] = ts
        logger.debug("\t continuous spike times")

        if tau_d > 0 and ax_x is not None:
            _exp = np.exp(-(t - t_spk) / tau_d)
            logger.debug("\t exp calculated")
            _decay = (synapse_mon.x_S[var_idx][::ss] - 1) * _exp
            logger.debug("\t decay calculated")
            _full_calc = mul * (1 + _decay)
            logger.debug("\t full y calculated")
            ax_x.plot(
                t / time_unit,
                _full_calc,
                linestyle="-",
                color=COLOR.CONN_dict[s_name],
                label=text.CONNECTION_MAP[s_name],
                rasterized=settings.RASTERIZED,
                **kwargs,
            )
            logger.debug("plotted ax_x reconstructed solutions")
        if tau_f > 0 and ax_u is not None:
            ax_u.plot(
                t / time_unit,
                mul * (synapse_mon.u_S[var_idx][::ss] * np.exp(-(t - t_spk) / tau_f)),
                linestyle="--",
                color=COLOR.CONN_dict[s_name],
                label=None,
                rasterized=settings.RASTERIZED,
                **kwargs,
            )

        if ax_w is not None:
            nspikes = np.sum(spk_index)
            x_S_spike = synapse_mon.x_S[var_idx][::ss][spk_index]
            u_S_spike = synapse_mon.u_S[var_idx][::ss][spk_index]
            lw = (
                kwargs["lw"]
                if "lw" in kwargs
                else kwargs["linewidth"]
                if "linewidth" in kwargs
                else 0.5
            )
            ax_w.vlines(
                t[spk_index] / time_unit,
                np.zeros(nspikes),
                mul * x_S_spike * u_S_spike / (1 - u_S_spike),
                color=COLOR.CONN_dict[s_name],
                lw=lw,
                rasterized=settings.RASTERIZED,
            )
            ax_w.plot(
                t[spk_index] / time_unit,
                mul * synapse_mon.dI_S[var_idx][::ss][spk_index],
                ".",
                color=COLOR.CONN_dict[s_name],
                rasterized=settings.RASTERIZED,
                **kwargs,
            )
    if ax_x and ax_u:
        xs_label = (
            text.VESICLES_LONG.replace("\n", "(%)\n")
            if perc
            else text.VESICLES_LONG
        )
        us_label = (
            text.EFFICACY_LONG.replace("\n", "(%)\n")
            if perc
            else text.EFFICACY_LONG
        )
    else:
        xs_label = text.VESICLES_TEXT
        us_label = text.EFFICACY_TEXT
        if perc:
            xs_label += " (%)"
            us_label += " (%)"
    w_label = (
        text.WEIGHT_LONG.replace("\n", "(%)\n") if perc else text.WEIGHT_LONG
    )
    if ax_x != ax_u:
        if ax_x is not None:
            ax_x.set_ylabel(xs_label)
        if ax_u is not None:
            ax_u.set_ylabel(us_label)
        if ax_w is not None:
            ax_w.set_ylabel(w_label)
    elif ax_x != ax_w:
        if ax_x is not None:
            ax_x.set_ylabel(f"{xs_label} \n {us_label}")
        if ax_w is not None:
            ax_w.set_ylabel(w_label)
    else:
        ax_x.set(ylabel=f"{xs_label} \n {us_label} \n {w_label}")

    ax_x.legend(loc=(0.0, 1.0), borderpad=0, ncol=4, frameon=False)


def plot_synaptic_var_zooms(
    *ax: Axes,
    xlims=None,
    width="100%",
    height="100%",
    xpad=0.1,
    ypad=0.04,
    time_unit=ms,
    **kwargs,
):
    """

    :param ax: Axes to zoom
    :param width: Size of the zoom (as string percentage of ax)
    :param time_unit: Time unit of plotting (for xlim)
    :return: Axis zoom(s)
    """
    ax_insets = []
    # calculate width and height per box
    if perc2frac(width) > 0.5 and xlims.__len__() > 1:
        width = frac2perc(perc2frac(width) / xlims.__len__())
    if ax.__len__() >= 1:
        g_height_f = perc2frac(height)
        height = frac2perc(g_height_f / ax.__len__())
        bbox_y = (g_height_f - ypad) / ax.__len__()
    for x_i, xlim in enumerate(xlims):
        bbox_start = 1.0 + x_i * (perc2frac(width) + xpad)
        for i, _ax in enumerate(ax):
            ax_inset = create_zoom(
                _ax,
                (width, height),
                loc="center left",
                xlim=xlim,
                xunit=defaultclock.dt / time_unit,
                inset_kwargs=dict(
                    bbox_to_anchor=(bbox_start, i * bbox_y + i * ypad, 1, 1),
                    bbox_transform=_ax.transAxes,
                ),
                box_kwargs=dict(zorder=99),
                **kwargs,
            )
            ax_inset.yaxis.tick_right()
            ax_inset.yaxis.set_label_position("right")
            label = _ax.get_ylabel()
            if "(" in label and ")" in label:
                _l_paren_idx = label.find("(")
                _r_paren_idx = label.find(")") + 1
                label = label[_l_paren_idx:_r_paren_idx]
            ax_inset.set_ylabel(label, color=COLOR.K)
            ax_insets.append(ax_inset)
    return ax_insets


def plot_diagram(
    N_E,
    N_I,
    ax=None,
    scale=10,
    e_color=COLOR.exc,
    i_color=COLOR.inh,
    cee=True,
    cei=True,
    cie=True,
    cii=True,
):
    from matplotlib.patches import Circle, Polygon, FancyArrowPatch

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.set_aspect(1.0)
    _x = ax.get_xlim()
    _y = ax.get_ylim()
    _x_unit = (_x[1] - _x[0]) / 4
    _y_unit = (_y[1] - _y[0]) / 4
    r = _y_unit / 2
    e_xy = (_x_unit, _y_unit * 2)
    e_xys = [
        [_x_unit - r, _y_unit * 2 - r],
        [_x_unit + r, _y_unit * 2 - r],
        [_x_unit, _y_unit * 2 + r],
    ]
    i_xy = (3 * _x_unit, _y_unit * 2)
    pop_e = Polygon(e_xys, color=e_color)
    pop_i = Circle(i_xy, radius=r, color=i_color)
    ax.add_artist(pop_e)
    ax.add_artist(pop_i)
    ax.text(_x_unit * 1, _y_unit * 2, N_E, fontsize="small", ha="center", va="center")
    ax.text(_x_unit * 3, _y_unit * 2, N_I, fontsize="small", ha="center", va="center")
    a = np.sqrt(np.power(r, 2) / 2)
    arrowprops_self = dict(connectionstyle="arc3,rad=-1.8", mutation_scale=scale)
    arrowprops_other = dict(connectionstyle="arc3,rad=-.5", mutation_scale=scale)
    if cee:
        c_e_e = FancyArrowPatch(
            posA=(e_xy[0] - a, e_xy[1] - a),
            posB=(e_xy[0] - a, e_xy[1] + a),
            ec=COLOR.C_E_E,
            arrowstyle="->",
            **arrowprops_self,
        )
        ax.add_artist(c_e_e)
    if cie:
        c_i_e = FancyArrowPatch(
            posA=(e_xy[0] + a, e_xy[1] + a),
            posB=(i_xy[0] - a, i_xy[1] + a),
            ec=COLOR.C_I_E,
            arrowstyle="->",
            **arrowprops_other,
        )
        ax.add_artist(c_i_e)
    if cei:
        c_e_i = FancyArrowPatch(
            posA=(i_xy[0] - a, i_xy[1] - a),
            posB=(e_xy[0] + a, e_xy[1] - a),
            ec=COLOR.C_E_I,
            arrowstyle="|-|,widthA=0.0,widthB=0.2",
            **arrowprops_other,
        )
        ax.add_artist(c_e_i)
    if cii:
        c_i_i = FancyArrowPatch(
            posA=(i_xy[0] + a, i_xy[1] + a),
            posB=(i_xy[0] + a, i_xy[1] - a),
            ec=COLOR.C_I_I,
            arrowstyle="|-|,widthA=0.0,widthB=0.2",
            **arrowprops_self,
        )
        ax.add_artist(c_i_i)
    for c in ax.get_children():
        c.set_clip_on(False)

    return fig, ax


def plot_raster_two_pop(N, N_E, sp_all, time_unit=ms, fig=None, ax=None):
    logger.debug("Raster plot")
    # Raster plot
    if fig is not None:
        fig_size = fig.get_size_inches()
        marker_size = fig_size[0] * fig_size[1] / (N * 4)
    else:
        marker_size = plt.rcParams["axes.markersize"]
    i, t = sp_all.it
    E_mask = i < N_E
    I_mask = i >= N_E
    plot_raster(
        i[E_mask],
        t[E_mask],
        time_unit,
        ax,
        color=COLOR.exc,
        marker=".",
        markersize=marker_size,
        rasterized=settings.RASTERIZED,
    )
    plot_raster(
        i[I_mask],
        t[I_mask],
        time_unit,
        ax,
        color=COLOR.inh,
        marker=".",
        markersize=marker_size,
        rasterized=settings.RASTERIZED,
    )
    ax.set_ybound(0, N)


def plot_state_colorbar(
    state_monitor,
    var_name,
    fig,
    ax,
    idx=None,
    label_text=True,
    label_coord=(-1.0, 0.0),
    cmap=None,
    time_unit=ms,
    extent=None,
    with_colorbar=False,
    colorbar_args=None,
    **kwargs,
):
    """Plot state value as a color-varying bar using `Axes.imshow` according to cmap.


    :param state_monitor: object with recorded variables. Must include var_name else an exception will be thrown by
        Brian2
    :type state_monitor: StateMonitor or pd.DataFrame
    :param var_name: variable recorded in `state_monitor`
    :type var_name: str
    :param fig: Figure
    :type fig: Figure
    :param ax: Axes
    :type ax: Axes
    :param idx: Numerical index for `state_monitor` (not neuron index)
    :param label_text: Add a label (default: var_name) to the colorbar according to `label_coord` or as the colorbar's
        label if `with_colorbar` is True
    :type label_text: str or bool
    :type label_coord: float
    :param cmap: Colormap for the axes.
        If a list of tuples is provided, a matplotlib.colors.LinearSegmentedColormap is made
    :type cmap: str or matplotlib.colors.Colormap or List[Tuple]
    :param time_unit: units for plotting
    :type time_unit: Quantity
    :param extent: Set the y limits to (bottom, top) by extending the drawn area of `imshow`
    :type extent: tuple or list
    :param with_colorbar: insert a colorbar
    :type with_colorbar: bool
    :param colorbar_args: arguments to pass to `Figure.colorbar`.  Properties overwrite `default_colorbar_props`
    :type colorbar_args: dict
    :param kwargs: keyword arguments to pass to `Axes.imshow`

    @return: instance of AxesImage or ColorbarPath (when `with_colorbar`=True)
    :rtype: AxesImage or ColorbarPath
    """
    if label_text is True:
        label_text = var_name

    resample_freq = int(1 / (defaultclock.dt / time_unit))  # use the same time_units
    arr = getattr(state_monitor, var_name)
    var_shape = arr.shape
    is_df = 1 if type(arr) is pd.DataFrame else 0
    if len(var_shape) == 1:
        var_shape = (0, var_shape[0])
    if idx is None or idx == -1:
        idx = (
            arr.columns
            if isinstance(arr, pd.DataFrame)
            else list(range(var_shape[is_df]))
        )
    T = int(var_shape[1 - is_df] / resample_freq)
    if extent is not None:
        extent = [0, T, extent[0], extent[1]]
    else:
        extent = [0, T, *ax.get_ylim()]
    if cmap is None:
        if "E_GABA" in var_name:
            _min = np.min(arr)
            _max = np.max(arr)
            cmap = LinearSegmentedColormap.from_list(
                "EGABA",
                [
                    (0, settings.COLOR.get_egaba_color(_min)),
                    (1, settings.COLOR.get_egaba_color(_max)),
                ],
            )
        else:
            cmap = LinearSegmentedColormap.from_list(
                var_name, [(0, "plum"), (1, "gray")]
            )
    elif np.iterable(cmap) and type(cmap[0]) is tuple:
        cmap = LinearSegmentedColormap.from_list(var_name, cmap)

    if np.iterable(idx) and len(idx):
        X = np.empty(shape=(len(idx), T))
        for i, ix in enumerate(idx):
            x = arr[ix][::resample_freq]
            if isinstance(x, Quantity):
                # noinspection PyProtectedMember
                x /= x.get_best_unit()
            elif isinstance(x, pd.Series):
                x = x.values
            X[i, :] = x
    elif np.iterable(idx):
        x = arr[::resample_freq]
        if isinstance(x, Quantity):
            # noinspection PyProtectedMember
            x /= x.get_best_unit()
        elif isinstance(x, pd.Series):
            x = x.values
        X = [x]
    else:
        x = arr[idx][::resample_freq]
        if isinstance(x, Quantity):
            # noinspection PyProtectedMember
            x /= x.get_best_unit()
        elif isinstance(x, pd.Series):
            x = x.values
        X = [x]
    im = ax.imshow(
        X=X, cmap=cmap, aspect="auto", interpolation="None", extent=extent, **kwargs
    )

    if with_colorbar:
        if colorbar_args is None:
            colorbar_args = dict(**default_colorbar_props)
        else:
            colorbar_args = dict(**default_colorbar_props, **colorbar_args)
        if with_colorbar == 2:
            # create colorbar without taking away any space
            cbar = colorbar_inset(im, **colorbar_args)
        else:
            # create colorbar taking away some space from ax
            cbar = fig.colorbar(im, ax, **colorbar_args)
        cbar.set_label(label_text, rotation=0, va="center", ha="left")
        return cbar
    elif label_text:
        if isinstance(label_coord, float):
            label_coord = (label_coord, 0.0)
        if label_coord[0] < 0:
            label_coord = (x.__len__() / 2, label_coord[1])
        ax.annotate(
            label_text, xy=label_coord, fontsize="x-small", ha="center", va="bottom"
        )
    return im


def plot_state_average(
    state_mon,
    variables=None,
    idxs=None,
    var_unit=None,
    var_names=None,
    ax=None,
    alpha=1.0,
    linestyles=None,
    lw=None,
    colors=None,
    blend=None,
    window=None,
    time_unit=ms,
    auto_legend=False,
    only_mean=False,
    **kwargs,
):
    """Plot variables in a state monitor, including the average across neurons for a given variable


    :type state_mon: StateMonitor or pd.DataFrame
    :param variables:
    :type variables: List[str] or str
    :param idxs: Array-relative indices to use (starting from 0 and ending at array size)
    :type idxs: List or ndarray
    :param var_unit: Value to be passed to plot_state in brian2tools
    :type var_unit: Quantity
    :param var_names: Values to be passed to plot_state in brian2tools
    :type var_names: List[str] or str
    :type ax: Axis
    :type alpha: List[float] or float
    :type linestyles: List[str] or str
    :param lw: shorthand for linewidth
    :type lw: List[float] or float
    :param colors:
    :type colors: List[str] or str
    :return: Axis used for plotting
    :rtype: Axis
    """
    if type(variables) is not list:
        variables = list(variables)
    if idxs is not None and type(idxs) is not np.ndarray:
        idxs = np.array(idxs)
    if not np.iterable(var_names):
        if var_names is None and "var_name" in kwargs:
            var_names = kwargs["var_name"]
            del kwargs["var_name"]
        var_names = [var_names] + [None] * (
            len(variables) - 1
        )  # use label once (cleaner legend)
    if type(alpha) is not list:
        alpha = [alpha] * len(variables)
    if type(linestyles) is not list:
        linestyles = [linestyles] * len(variables)
    if type(lw) is not list:
        if lw is None:
            lw = plt.rcParams["axes.linewidth"]
        lw = [lw] * len(variables)
    if not np.iterable(colors):
        if colors is None and "color" in kwargs:
            colors = kwargs["color"]
            del kwargs["color"]
        colors = [colors] * len(variables)
    elif isinstance(colors, str):
        colors = [colors] * len(variables)

    # we plot the mean first for legend purposes
    for v, variable in tqdm(
        enumerate(variables),
        desc="Plotting state average",
        # disable=logging.getLogger().getEffectiveLevel() > logging.INFO,
        leave=True,
    ):
        view = getattr(state_mon, variable)
        if idxs is not None:
            view = view[idxs]
        view_mean = view.mean(axis=int(isinstance(view, pd.DataFrame)))
        if not np.iterable(view_mean):
            view_mean = view.values
        if window is not None:
            view_mean = (
                pd.DataFrame(view_mean)
                .rolling(int(window / defaultclock.dt))
                .mean()
                .T.values[0]
            )
        ax = plot_state(
            state_mon.t,
            view_mean,
            axes=ax,
            var_unit=var_unit,
            time_unit=time_unit,
            var_name=var_names[v],
            linestyle=linestyles[v],
            lw=lw[v],
            color=colors[v],
            alpha=alpha[v],
            rasterized=settings.RASTERIZED,
            label=var_names[v],
            **kwargs,
        )
    if not only_mean and getattr(state_mon, variables[0]).shape[0] > 1:
        with tqdm(
            len(variables),
            desc="Plotting state",
            disable=logging.getLogger().getEffectiveLevel() > logging.INFO,
            leave=False,
        ) as pbar:
            for v, variable in enumerate(variables):
                pbar.set_description(f"Plotting state {variable}")
                view = getattr(state_mon, variable)
                if idxs is not None:
                    view = view[idxs]
                if window is not None:
                    view = (
                        pd.DataFrame(view)
                        .rolling(int(window / defaultclock.dt), axis=1)
                        .mean()
                        .values
                    )
                if blend is not None:
                    base_color = spectra.html(colors[v])
                    for i, blend_c in enumerate(blend):
                        if blend_c in ["E", "I"]:
                            blend_c = COLOR.CONN_BLEND[blend_c]
                        color = base_color.blend(spectra.html(blend_c), 0.75).hexcode
                        plot_state(
                            state_mon.t,
                            view[i],
                            axes=ax,
                            var_unit=var_unit,
                            time_unit=time_unit,
                            var_name=var_names[v],
                            linestyle=linestyles[v],
                            lw=lw[v] / 2,
                            color=color,
                            alpha=alpha[v] / 2,
                            label=f"{var_names[v]}_{v} {blend_c}",
                            rasterized=settings.RASTERIZED,
                            **kwargs,
                        )
                    # if len(blend) == 2: if uncommented, need to reorder correctly
                    #     ax.fill_between(state_mon.t / time_unit,
                    #                     view[0] / var_unit, view[1] / var_unit,
                    #                     color=color, alpha=alpha[v]/4,
                    #                     zorder=-99999)
                else:
                    color = colors[v]
                    plot_state(
                        state_mon.t,
                        view if isinstance(view, pd.DataFrame) else view.T,
                        axes=ax,
                        var_unit=var_unit,
                        time_unit=time_unit,
                        var_name=var_names[v],
                        linestyle=linestyles[v],
                        lw=lw[v] / 2,
                        color=color,
                        alpha=alpha[v] / 4,
                        label=f"{var_names[v]}_{v}",
                        rasterized=settings.RASTERIZED,
                        **kwargs,
                    )
                pbar.update(1)
    if auto_legend:
        if type(auto_legend) is dict:
            ax.legend(**auto_legend)
        elif type(auto_legend) is list:
            ax.legend(*auto_legend)
        else:
            ax.legend()
    return ax


def isi(N, N_E, N_I, duration, static_cl_dt, sp_all):
    logger.info("ISI")
    _, ax_isi = plt.subplots()
    sp_trains = sp_all.spike_trains()
    for col, n_neuron_samples in zip(
        [COLOR.exc, COLOR.inh, COLOR.average],
        [(0, N_E), (N_E, N_I), (0, N)],
    ):
        logger.info("COLOR = {}".format(col))
        n_ecl_jumps = int(duration / static_cl_dt)
        ecl_increments = np.linspace(
            0 * second, int(duration + static_cl_dt) * second, n_ecl_jumps + 1
        )
        ecl_increments_plot = np.linspace(
            0 * second, int(duration + static_cl_dt) * ms, n_ecl_jumps
        )
        isi_mu = np.full((n_neuron_samples, n_ecl_jumps), np.nan) * second
        isi_std = np.full((n_neuron_samples, n_ecl_jumps), np.nan) * second
        for nrn_idx in np.arange(*n_neuron_samples):
            train_sample = sp_trains[nrn_idx]
            for idx in range(1, n_ecl_jumps):
                bool_values = (ecl_increments[idx - 1] < train_sample) & (
                    train_sample < ecl_increments[idx]
                )
                train = np.diff(train_sample[bool_values])
                if len(train) > 1:
                    isi_mu[nrn_idx, idx] = np.mean(train)
                    isi_std[nrn_idx, idx] = np.std(train)
            # ax_isi.errorbar(ecl_increments / ms, isi_mu[nrn_idx] / ms, yerr=isi_std[nrn_idx] / ms, color=colors[
            # nrn_idx])
        isi_mu_mean = np.nanmean(isi_mu, axis=0)
        isi_std_mean = np.nanmean(isi_std, axis=0)
        ax_isi.errorbar(
            ecl_increments_plot / ms,
            isi_mu_mean / ms,
            yerr=isi_std_mean / ms,
            color=col,
            alpha=0.5,
        )
    ax_isi.set_xlabel("Time (ms)")
    ax_isi.set_ylabel("Interspike interval (ms)")


def visualise_connectivity(
    N, N_E, N_I, C_E_E, C_I_E, C_E_I, C_I_I, p_E_E, p_I_E, p_E_I, p_I_I, plot_type=None
):
    logger.info("Connectivity (scatter)")
    from typing import List

    ax_con: List[Axes]
    fig_con, ax_con = plt.subplots(nrows=1, ncols=1)
    ax_con = [ax_con]
    figsize = fig_con.get_size_inches()
    plot_type = plot_type or ("scatter" if N <= 1000 else "hexbin")
    markersize = figsize[0] * figsize[1] / 1000 if plot_type == "scatter" else None
    plot_synapses_kwargs = dict()
    if plot_type == "scatter":
        plot_synapses_kwargs["s"] = markersize
        # fig_synapses, ax = subplots()
        plot_synapses(
            C_E_E.i,
            C_E_E.j,
            var_name="E->E",
            axes=ax_con[0],
            **plot_synapses_kwargs,
            color=COLOR.CONN_dict["C_E_E"],
        )
        plot_synapses(
            C_I_E.i,
            C_I_E.j + N_E,
            var_name="I->E",
            axes=ax_con[0],
            **plot_synapses_kwargs,
            color=COLOR.CONN_dict["C_I_E"],
        )
        plot_synapses(
            C_E_I.i + N_E,
            C_E_I.j,
            var_name="E->I",
            axes=ax_con[0],
            **plot_synapses_kwargs,
            color=COLOR.CONN_dict["C_E_I"],
        )
        plot_synapses(
            C_I_I.i + N_E,
            C_I_I.j + N_E,
            var_name="I->I",
            axes=ax_con[0],
            **plot_synapses_kwargs,
            color=COLOR.CONN_dict["C_I_I"],
        )
    else:
        logger.info("Connectivity (hexbin)")
        sources = np.concatenate([C_E_E.i, C_I_E.i, C_E_I.i + N_E, C_I_I.i + N_E])
        targets = np.concatenate([C_E_E.j, C_I_E.j + N_E, C_E_I.j, C_I_I.j + N_E])
        plot_synapses(sources, targets, plot_type="hexbin", axes=ax_con[1])
    ax_con[0].annotate("E->E ({})".format(p_E_E), (N_E / 2, N_E / 2))
    ax_con[0].annotate("E->I ({})".format(p_I_E), (N_E + N_I / 2, N_E / 2))
    ax_con[0].annotate("I->E ({})".format(p_E_I), (N_E / 2, N_E + N_I / 2))
    ax_con[0].annotate("I->I ({})".format(p_I_I), (N_E + N_I / 2, N_E + N_I / 2))
    ax_con[0].axhline(N_E, lw=1, color="k")
    ax_con[0].axvline(N_E, lw=1, color="k")
    return fig_con, ax_con


def add_mg_ax(self, ax: Axes, time_unit=ms, append_axes=False):
    divider = None
    if append_axes:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        ax = divider.append_axes("top", size="15%", pad=0.0, sharex=ax)
        adjust_spines(ax, [], 0)
    x_coord = (
        self.zero_mag_off_t - self.zero_mag_onset_t
    ) / 2 + self.zero_mag_wash_rate
    im = plot_state_colorbar(
        self.mg2_mon,
        "Mg2",
        fig=ax.figure,
        ax=ax,
        time_unit=time_unit,
        label_text="$0 [Mg^{2+}]$",
        label_coord=x_coord / time_unit,
    )
    return im, divider
