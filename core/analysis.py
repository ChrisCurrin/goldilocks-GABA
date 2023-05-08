# coding=utf-8
"""Methods to analyse completed simluations"""
import logging

import brian2.numpy_ as np
import pandas as pd
from brian2 import PopulationRateMonitor, defaultclock
from brian2.units import Hz, Quantity, Unit, ms, second
from matplotlib import pyplot as plt

logger = logging.getLogger("analysis")

# suppress numpy warnings
np.seterr(all="ignore")


def spikes_to_rate(spk_mon, time_unit=ms, bin_width=10 * ms):
    """
    Convert spikes from a SpikeMonitor to a rate (in pandas Series)

    Args:
        spk_mon: SpikeMonitor
        time_unit: time unit of the rate
        bin_width: width of the bin (in time_unit)

    Returns:
        rate: pandas Series of the rate
    """
    n_indices, t_indices = spk_mon.it
    duration = t_indices[-1]
    N = max(n_indices) + 1
    spk_count, bin_edges = np.histogram(
        t_indices / time_unit, int(duration / bin_width)
    )
    _rate = np.double(spk_count) / N / bin_width / Hz
    _time = np.arange(bin_width, duration, bin_width)
    return pd.Series(_rate, index=_time)


def burst_stats(
    rate_monitor: PopulationRateMonitor,
    xlim=None,
    width=50.1 * ms,
    time_unit=ms,
    rate_std_thresh=2.0,
    rate_thresh=20.0 * Hz,
    time_thresh=20 * ms,
    plot_fig=True,
):
    """Get the burst start and end times.

    Burst detector is where rate is a std deviation above the mean, so works best when the length of the burst is
    much shorter than the time between bursts to get a low mean.

    Args:
        rate_monitor: PopulationRateMonitor or Series or DataFrame or tuple of rate and time
        xlim: tuple of start and end time
        width: width of the smoothing window
        time_unit: time unit of the rate_monitor
        rate_std_thresh: threshold for the rate in std deviations above the mean
        rate_thresh: threshold for the rate in Hz
        time_thresh: threshold for the time in ms
        plot_fig: plot the rate and burst start and end times

    Returns:
        burst_start_ts: burst start times
        burst_end_ts: burst end times

    """
    from scipy import stats

    logger.debug("burst_stats")
    if isinstance(rate_monitor, pd.Series):
        _rate = rate_monitor.values
        _time = rate_monitor.index.values
    elif isinstance(rate_monitor, pd.DataFrame):
        _rate = rate_monitor.smooth_rate(None, None)
        _time = rate_monitor.index.values
    elif np.iterable(rate_monitor):
        _rate = rate_monitor[0]
        _time = rate_monitor[1]
    else:
        _rate = rate_monitor.smooth_rate(window="flat", width=width) / Hz
        _time = rate_monitor.t / time_unit
    if xlim is not None:
        _x0, _x1 = xlim
        _rate = _rate[_x0:_x1]
        _time = _time[_x0:_x1]
    descriptive_stats = stats.describe(_rate)
    variance = descriptive_stats.variance
    mu = descriptive_stats.mean
    std_dev = np.sqrt(variance)
    burst_detector = np.logical_and(
        _rate > mu + std_dev * rate_std_thresh, _rate > rate_thresh / Hz
    )
    if not np.any(burst_detector):
        # no bursts detected (all values False)
        burst_start_times, burst_end_times = np.array([]), np.array([])
    else:
        bursts_t = _time[burst_detector]
        # bursts_y = _rate[burst_detector]

        dt_mask = np.diff(bursts_t) > 1

        # detect start by sharp gradient
        burst_end_times = np.append(bursts_t[:-1][dt_mask], bursts_t[-1])
        burst_start_times = np.append(bursts_t[0], bursts_t[1:][dt_mask])

        burst_durations = burst_end_times - burst_start_times
        mask = burst_durations > time_thresh / time_unit
        burst_start_times = burst_start_times[mask]
        burst_end_times = burst_end_times[mask]
        burst_durations, inter_burst_intervals = get_duration_interval(
            burst_start_times, burst_end_times
        )
        burst_duration = np.mean(burst_durations)
        inter_burst_interval = np.mean(inter_burst_intervals)
        logger.debug("\t burst_duration = {}".format(burst_duration))
        logger.debug("\t inter_burst_interval = {}".format(inter_burst_interval))
    if plot_fig:
        kwargs = {
            "lw": 0.5,
            "color": "k",
            "burst_kwargs": {"alpha": 0.5, "lw": 0.5, "ms": 0.1, "color": "g"},
            "plot_burst_duration": False,
        }
        if isinstance(plot_fig, plt.Axes):
            ax: plt.Axes = plot_fig
        elif isinstance(plot_fig, dict):
            if "ax" not in plot_fig:
                fig, ax = plt.subplots()
            else:
                ax = plot_fig.pop("ax")
            kwargs.update(plot_fig)
        else:
            fig, ax = plt.subplots()
        burst_kwargs = kwargs.pop("burst_kwargs", {})
        plot_burst_duration = kwargs.pop("plot_burst_duration", False)
        ax.plot(_time, _rate, **kwargs)
        y_top = np.max(_rate) + 5
        if burst_kwargs:
            if plot_burst_duration:
                for st, et in zip(burst_start_times, burst_end_times):
                    ax.plot([st, et], [y_top] * 2, **burst_kwargs)
            ax.plot(
                burst_start_times, [y_top] * len(burst_start_times), ">", **burst_kwargs
            )
        ax.set_ylim(0)

    return burst_start_times, burst_end_times


def get_duration_interval(burst_start_times, burst_end_times):
    burst_durations = burst_end_times - burst_start_times
    inter_burst_intervals = burst_start_times[1:] - burst_end_times[:-1]
    return burst_durations, inter_burst_intervals


def get_burst_periods(burst_start_times, burst_end_times):
    """Create different x-lim windows based on the start and end times of bursts.

    The returned x-lim windows are
        - 'burst': from burst start to burst end
        - 'interburst': from burst end to burst start (ignores from start to first burst in domain)
        - 'full': from burst start to next burst start (does not have last burst in domain)
        - 'mid': centers the burst n with domain from middle of interburst n to middle of interburst n+1.

    Args:
        burst_start_times (np.ndarray): start times of bursts
        burst_end_times (np.ndarray): end times of bursts

    Returns:
        dict: dictionary of x-lim windows for each burst type (burst, interburst, full, mid)
            as a list of tuples


    """
    burst_data = {
        "burst": [],
        "interburst": [],
        "full": [],
        "mid": [],
    }
    if burst_start_times.__len__() == 0:
        return burst_data
    elif burst_start_times.__len__() == 1:
        interburst_periods = [(0, burst_start_times[0])]
        full_periods = [(0, burst_end_times[0])]
    else:
        interburst_periods = list(zip(burst_end_times, burst_start_times[1:]))
        full_periods = list(zip(burst_start_times, burst_start_times[1:]))
    burst_periods = list(zip(burst_start_times, burst_end_times))
    interburst_interval = (
        burst_start_times[1:] - burst_end_times[:-1]
    )  # get durations between bursts
    half_interval = interburst_interval / 2
    # burst_dur = burst_end_times - burst_start_times  # get durations of bursts
    # half_d = burst_dur / 2
    # mid_burst = burst_start_times + half_d
    mid_period_times = burst_start_times[1:] - half_interval
    try:
        # include first burst
        mid_period_times = np.insert(
            mid_period_times, 0, burst_start_times[0] - half_interval[0]
        )
    except IndexError:
        pass
    mid_periods = list(zip(mid_period_times, mid_period_times[1:]))
    burst_data["burst"] = burst_periods
    burst_data["interburst"] = interburst_periods
    burst_data["full"] = full_periods
    burst_data["mid"] = mid_periods

    return burst_data


def get_x_idxs(*x_vals: Unit, time_unit=ms):
    """Convert time unit-based x values to indices"""
    if np.iterable(x_vals[0]):
        x_vals = x_vals[0]
    return (x_vals * time_unit / defaultclock.dt).astype(int)


def inst_burst_rate(
    burst_times, T, window=30 * second, rolling=30 * second, time_unit=ms
):
    """
    Calculate instantaneous burst rate for a given time window and rolling window.

    Args:
        burst_times (list[float]): list of burst times
        T (float): total time of simulation
        window (float): window size for calculating instantaneous burst rate
        rolling (float): rolling window size for calculating instantaneous burst rate
        time_unit (float): time unit of simulation

    Returns:
        list[float]: list of instantaneous burst rates

    """
    n_windows = (
        np.round(T / rolling)
        if isinstance(T, Quantity)
        else np.round(T * time_unit / rolling)
    )
    t_points = [np.round(i * rolling / time_unit) for i in range(int(n_windows))]
    n_bursts = []
    for t_point in t_points:
        accum = 0
        for s_t in burst_times:
            if t_point < s_t < t_point + window / time_unit:
                accum += 1
        n_bursts.append(accum)
    # alt method
    result2 = []
    for t_point in t_points:
        result2.append(
            np.sum(
                (
                    burst_times[
                        np.logical_and(
                            t_point <= burst_times,
                            burst_times < t_point + window / time_unit,
                        )
                    ]
                )
                > 0
            ).astype(int)
        )
    logger.debug(f"\n\tn_bursts = {n_bursts} \n\t result_alt = {result2}")
    return t_points, n_bursts
