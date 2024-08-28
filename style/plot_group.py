# coding=utf-8
"""
Helper visualisation methods
"""
import logging

import brian2.numpy_ as np
import seaborn as sns
from brian2.units import Quantity, ms, mV, nS, pA, uS
from brian2tools import plot_rate
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import settings
from settings import COLOR, text
from style.axes import adjust_spines, align_axes
from style.plot_trace import (
    plot_raster_two_pop,
    plot_spectrogram,
    plot_state_average,
    plot_state_colorbar,
    spike_train_profile,
)

logger = logging.getLogger("viz")
plot_fig = True
save_fig = False
clear_all = False
file_formats = None
time_unit = ms


def activity_plot(
    N,
    N_E,
    sp_all,
    r_all,
    r_E,
    r_I,
    mg2_mon,
    state_mon,
    zero_mag_onset_t=-1,
    zero_mag_off_t=-1,
    zero_mag_wash_rate=0.0,
    time_unit=ms,
    ax=None,
    fig=None,
):
    """Plot activity of the group of neurons"""
    logger.info("plotting activity")
    # ###########################################
    # Make plots
    # ###########################################

    is_new_ax = ax is None
    if ax is None:
        height_ratios = [
            10,  # raster plot
            6,  # population rate
            3,  # spectrogram
            3,  # spike train profile 1
            3,  # spike train profile 2
        ]
        fig, ax = plt.subplots(
            nrows=len(height_ratios),
            ncols=1,
            sharex="col",
            gridspec_kw={"height_ratios": height_ratios},
        )
    else:
        assert fig is not None

    # plot Mg2
    divider = make_axes_locatable(ax[0])
    mg_ax = divider.append_axes("top", size="10%", pad=0.01, sharex=ax[-1])
    ax = np.insert(ax, 0, mg_ax)
    x_coord = (zero_mag_off_t - zero_mag_onset_t) / 2 + zero_mag_wash_rate
    plot_state_colorbar(
        mg2_mon,
        "Mg2",
        fig=fig,
        ax=mg_ax,
        idx=0,
        label_text="0 $Mg^{2+}$",
        label_coord=x_coord / time_unit,
        time_unit=time_unit,
    )

    adjust_spines(ax[0], [])
    if sp_all is not None:
        plot_raster_two_pop(N, N_E, sp_all, time_unit=time_unit, fig=fig, ax=ax[1])
    if r_all is not None:
        logger.info("Population rates")
        # Population rates
        smooth_rate_args = dict(window="flat", width=10.1 * ms)
        for rate_mon, col in zip([r_E, r_I, r_all], settings.COLOR.EIA_list):
            plot_rate(
                rate_mon.t,
                rate_mon.smooth_rate(**smooth_rate_args),
                time_unit=time_unit,
                axes=ax[2],
                linewidth=1,
                color=col,
                alpha=0.6,
                rasterized=settings.RASTERIZED,
            )

        leg_texts = []
        for rate_mon, col, leg_text in zip(
            [r_E, r_I, r_all],
            settings.COLOR.EIA_list,
            ["Excitatory", "Inhibitory", "Average"],
        ):
            # mean and std deviation
            mean_rate = np.mean(rate_mon.rate)
            # std_rate = std(rate_mon.rate / Hz)
            ax[2].axhline(
                mean_rate,
                color=col,
                lw=1,
                alpha=0.5,
                linestyle="--",
                rasterized=settings.RASTERIZED,
            )
            #     ax[1].axhline(mean_rate + std_rate, color=col, lw=1, alpha=0.25, linestyle=':')
            ax[2].annotate(
                "{:.2f}".format(mean_rate), xy=(0, mean_rate), color=col, ha="left"
            )
            leg_texts.append("{}({:.2f})".format(leg_text, mean_rate))
        ax[2].legend(
            leg_texts,
            # bbox_to_anchor=(1, 0), loc=3
        )
        # Spectrogram
        plot_spectrogram(
            r_all.smooth_rate(**smooth_rate_args),
            ax=ax[3],
            time_unit=time_unit,
            Fs=smooth_rate_args["width"] / ms,
        )
    if sp_all is not None:
        spike_train_profile(sp_all, ax=ax[4:6], time_unit=time_unit)
        ax[-1].set_xbound(0, sp_all.t[-1] / time_unit)

    logger.info("Adjust")
    for _ax in ax[:-1]:
        sns.despine(ax=_ax, top=True, right=True, left=False, bottom=True)
        _ax.set_xlabel("")
    sns.despine(ax=ax[-1], top=True, right=True, left=False, bottom=False)
    ax[-1].set_xlabel("Time (s)")
    align_axes(ax)
    return fig, ax


def simple_activity_plot(r_all, r_E, r_I, state_mon, time_unit=ms, ax=None, fig=None):
    """Plot activity of the group of neurons"""
    logger.info("plotting activity")
    # ###########################################
    # Make plots
    # ###########################################

    # noinspection PyTypeChecker
    new_ax = ax is None
    if ax is None:
        height_ratios = [1, 10]
        fig, ax = plt.subplots(
            nrows=len(height_ratios),
            ncols=1,
            sharex="col",
            gridspec_kw={"height_ratios": height_ratios},
        )
    else:
        assert fig is not None

    # plot Mg2
    cmap = LinearSegmentedColormap.from_list(
        "E_GABA", [(0, settings.COLOR.inh), (1, "plum")]
    )
    # divider = make_axes_locatable(ax[0])
    # egaba_ax = divider.append_axes("top", size='10%', pad=0.01, sharex=ax[-1])
    # ax = insert(ax, 0, egaba_ax)
    egaba_ax = ax[0]
    plot_state_colorbar(
        state_mon,
        "E_GABA",
        fig,
        egaba_ax,
        idx=0,
        time_unit=time_unit,
        label_text=f"depolarising {text.EGABA}",
        label_coord=state_mon.t[-1] * 3 / 4 / time_unit,
        cmap=cmap,
    )
    egaba_ax.annotate(
        f"hyperpolarising {text.EGABA}",
        xy=(state_mon.t[-1] / 4 / time_unit, 0),
        fontsize="x-small",
        ha="center",
        va="center",
    )
    sns.despine(ax=ax[0], top=True, right=True, left=True, bottom=True)

    logger.info("Population rates")
    # Population rates
    smooth_rate_args = dict(window="flat", width=10.1 * ms)
    for rate_mon, col in zip([r_E, r_I], [settings.COLOR.exc, settings.COLOR.inh]):
        plot_rate(
            rate_mon.t,
            rate_mon.smooth_rate(**smooth_rate_args),
            time_unit=time_unit,
            axes=ax[1],
            linewidth=1,
            color=col,
            alpha=0.6,
            rasterized=settings.RASTERIZED,
        )

    ax[1].legend(["Excitatory", "Inhibitory"])

    ax[-1].set_xbound(0, state_mon.t[-1] / time_unit)

    logger.info("Adjust")
    for _ax in ax[:-1]:
        sns.despine(ax=_ax, top=True, right=True, left=False, bottom=True)
        _ax.set_xlabel("")
    sns.despine(ax=ax[-1], top=True, right=True, left=False, bottom=False)
    ax[-1].set_xlabel("Time (s)")
    align_axes(ax)
    return fig, ax


def plot_hierarchy(
    N,
    N_E,
    sp_all,
    r_all,
    r_E,
    r_I,
    state_mon,
    V_thr,
    nrn_idx=None,
    time_unit=ms,
    ax=None,
    fig=None,
):
    logger.info("plotting hierarchy")
    # ###########################################
    # Make plots
    # ###########################################

    is_new_ax = ax is None
    if ax is None:
        height_ratios = [
            1,  # synapse
            1,  # neuron
            1,  # raster
            1,  # population rate
        ]
        fig, ax = plt.subplots(
            nrows=len(height_ratios),
            ncols=1,
            sharex="col",
            figsize=(settings.PAGE_W_FULL, settings.PAGE_H_FULL),
            gridspec_kw={"height_ratios": height_ratios},
        )
    else:
        assert fig is not None

    ax_syn, ax_neuron, ax_raster, ax_pop = ax

    var_attrs = {
        "v": {"name": "V", "color": COLOR.K, "lw": 0.1},
        "E_Cl": {"name": text.ECL, "color": COLOR.G, "lw": 0.5},
        "E_GABA": {"name": text.EGABA, "color": COLOR.R1_B3, "lw": 1.5},
    }

    #
    # Postsynaptic conductances
    #
    logger.info("Postsynaptic conductances")
    plot_state_average(
        state_mon,
        ["g_NMDA", "g_AMPA", "g_GABA"],
        var_unit=nS,
        ax=ax_syn,
        alpha=0.8,
        colors=[COLOR.exc_alt, COLOR.exc, COLOR.inh],
        # linestyles=['-'],
        only_mean=True,
        lw=0.1,
        time_unit=time_unit,
        # window=10.1*ms,
    )

    ax_syn.axhline(color=COLOR.K, alpha=0.2, lw=0.5, linestyle=":")

    sum_g_NMDA = np.sum(state_mon.g_NMDA, axis=1)
    sum_g_AMPA = np.sum(state_mon.g_AMPA, axis=1)
    sum_g_GABA = np.sum(state_mon.g_GABA, axis=1)

    weighted = sum_g_NMDA + sum_g_AMPA - sum_g_GABA
    g_vars = {
        "$g_{NMDA}$": sum_g_NMDA,
        "$g_{AMPA}$": sum_g_AMPA,
        "$g_{GABA}$": sum_g_GABA,
        r"$\sum $": weighted,
    }
    legend_vals = []
    for g_var_k, g_var_v in g_vars.items():
        _g_str = ", ".join(["{:>6.2f}".format(g_var / uS) for g_var in g_var_v])
        legend_vals.append(f"{g_var_k} = {_g_str} (uS)")

    ax_syn.legend(
        ["$g_{NMDA}$", "$g_{AMPA}$", "$g_{GABA}$"],
        frameon=False,
        # , bbox_to_anchor=(1, 0), loc=3
    )
    # ax[0].axhline(weighted, lw=0.5, color=COLOR.R2_B1, linestyle='--')
    # Adjust axis
    ax_syn.set(ylabel="postsynaptic\nconductance\n(nS)")

    # #
    # # Membrane potential
    # #
    plot_state_average(
        state_mon,
        ["v"],
        var_names=var_attrs["v"]["name"],
        # idxs=nrn_idx,
        ax=ax_neuron,
        alpha=0.8,
        colors=[var_attrs["v"]["color"]],
        lw=var_attrs["v"]["lw"],
        time_unit=time_unit,
    )
    if sp_all is not None:
        logger.info("Membrane potential")
        ax_neuron.axhline(V_thr / mV, color="Grey", linestyle=":", lw=0.5)  # Threshold
        # Artificially insert spikes
        for i, _ in enumerate(nrn_idx):
            ax_neuron.vlines(
                sp_all.t[sp_all.i == nrn_idx[i]] / time_unit,
                V_thr / mV,
                0,
                color=COLOR.K,
                lw=var_attrs["v"]["lw"],
                alpha=0.8,
            )
    ax_neuron.set_ylabel("membrane\npotential\n(mV)")
    if sp_all is not None:
        plot_raster_two_pop(N, N_E, sp_all, time_unit=time_unit, fig=fig, ax=ax_raster)

    if r_all is not None:
        logger.info("Population rates")
        # Population rates
        smooth_rate_args = dict(window="flat", width=10.1 * ms)
        # for rate_mon, col in zip([r_E, r_I, r_all], settings.COLOR.EIA_list):
        #     plot_rate(rate_mon.t, rate_mon.smooth_rate(**smooth_rate_args), time_unit=time_unit,
        #               axes=ax_pop, linewidth=1, color=col, alpha=0.6, rasterized=settings.RASTERIZED)
        plot_rate(
            r_all.t,
            r_all.smooth_rate(**smooth_rate_args),
            time_unit=time_unit,
            axes=ax_pop,
            linewidth=0.5,
            color=settings.COLOR.K,
            alpha=1,
            rasterized=settings.RASTERIZED,
        )
        # ax_pop.legend(["Excitatory population", "Inhibitory population", "Average"], frameon=False,)
    logger.info("Adjust")
    for _ax in ax[:-1]:
        sns.despine(ax=_ax, top=True, right=True, left=False, bottom=True)
        _ax.set_xlabel("")
    sns.despine(ax=ax[-1], top=True, right=True, left=False, bottom=False)
    ax[-1].set_xbound(0, r_all.t[-1] / time_unit)
    align_axes(ax)
    ax[-1].set_xlabel("Time (s)")
    return fig, ax


def plot_states(
    state_mon,
    synapse_monitors,
    V_thr,
    V_reset,
    sp_all,
    tau_d,
    tau_f,
    nrn_idx=None,
    nrn_idx_type=None,
    fig=None,
    ax=None,
    time_unit=ms,
):
    #
    # Dynamics of a single neuron
    #
    logger.info("plotting states for neuron idx_{}".format(nrn_idx))
    new_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(
            nrows=5,
            ncols=1,
            sharex=True,
            gridspec_kw={
                "height_ratios": [
                    5,  # voltage
                    2,  # inhibition
                    3,  # g
                    5,  # synaptic variables
                    1,
                ],  # dynamic weight
            },
        )
    else:
        assert fig is not None
    ax_v, ax_inh, ax_ps_g, ax_syn_var, ax_w = ax

    var_attrs = {
        "v": {"name": "V", "color": COLOR.K, "lw": 0.5},
        "E_Cl": {"name": text.ECL, "color": COLOR.G, "lw": 0.5},
        "E_GABA": {"name": text.EGABA, "color": COLOR.R1_B3, "lw": 1.5},
    }

    # #
    # # Membrane potential
    # #
    plot_state_average(
        state_mon,
        ["v"],
        var_names=var_attrs["v"]["name"],
        ax=ax_v,
        alpha=0.8,
        colors=var_attrs["v"]["color"],
        lw=var_attrs["v"]["lw"],
        time_unit=time_unit,
    )
    if sp_all is not None:
        logger.info("Membrane potential")
        ax_v.axhline(V_thr / mV, color="Grey", linestyle=":", lw=0.5)  # Threshold
        # Artificially insert spikes
        for i, _ in enumerate(nrn_idx):
            ax_v.vlines(
                sp_all.t[sp_all.i == nrn_idx[i]] / time_unit,
                V_thr / mV,
                0,
                color=COLOR.K,
                lw=0.5,
                alpha=0.8,
            )
    #
    # Reversal Potential for Chloride over time
    #
    logger.info("ECl over time")
    plot_vars = ["E_Cl", "E_GABA"]
    actual_vars = []
    var_names = []
    var_colors = []
    var_lw = []
    for _var in plot_vars:
        if hasattr(state_mon, _var):
            actual_vars.append(_var)
            var_names.append(var_attrs[_var]["name"])
            var_colors.append(var_attrs[_var]["color"])
            var_lw.append(var_attrs[_var]["lw"])
    plot_state_average(
        state_mon,
        actual_vars,
        var_names=var_names,
        ax=ax_inh,
        alpha=0.8,
        colors=var_colors,
        lw=var_lw,
        time_unit=time_unit,
    )

    ax_inh.legend(["$V_m$", "$E_{Cl}$", "$E_{GABA}$"])
    # plot E for bicarb
    # ax_mem.axhline(-10, color=COLOR.R1_B2, alpha=0.5, lw=0.5, linestyle='--')

    i_ax4 = ax_inh.twinx()
    plot_state_average(
        state_mon,
        ["I_GABA_rec"],
        var_unit=pA,
        var_names=["$I_{GABA}$"],
        ax=i_ax4,
        alpha=0.8,
        colors=[COLOR.inh],
        lw=[0.5],
        time_unit=time_unit,
    )
    # i_ax4.axhline(color=COLOR.K, alpha=0.2, lw=0.5, linestyle=':')

    adjust_spines(i_ax4, ["right"], sharedx=True)

    #
    # Postsynaptic conductances
    #
    logger.info("Postsynaptic conductances")
    plot_state_average(
        state_mon,
        ["g_NMDA", "g_AMPA", "g_GABA"],
        var_unit=nS,
        ax=ax_ps_g,
        alpha=0.8,
        colors=[COLOR.exc_alt, COLOR.exc, COLOR.inh],
        blend=nrn_idx_type,
        linestyles=["--", ":", "-"],
        lw=0.5,
        time_unit=time_unit,
        window=100 * ms,
    )

    ax_ps_g.axhline(color=COLOR.K, alpha=0.2, lw=0.5, linestyle=":")

    sum_g_NMDA = np.sum(state_mon.g_NMDA, axis=1)
    sum_g_AMPA = np.sum(state_mon.g_AMPA, axis=1)
    sum_g_GABA = np.sum(state_mon.g_GABA, axis=1)

    weighted = sum_g_NMDA + sum_g_AMPA - sum_g_GABA
    g_vars = {
        "$g_{NMDA}$": sum_g_NMDA,
        "$g_{AMPA}$": sum_g_AMPA,
        "$g_{GABA}$": sum_g_GABA,
        "$\sum $": weighted,
    }
    legend_vals = []
    for g_var_k, g_var_v in g_vars.items():
        legend_vals.append(
            g_var_k
            + "$="
            + ", ".join(["{:>6.2f}".format(g_var / uS) for g_var in g_var_v])
            + "uS$"
        )
    ax_ps_g.legend(
        ["$g_{NMDA}$", "$g_{AMPA}$", "$g_{GABA}$"]
        # , bbox_to_anchor=(1, 0), loc=3
    )
    # ax[0].axhline(weighted, lw=0.5, color=COLOR.R2_B1, linestyle='--')
    # Adjust axis
    ax_ps_g.set(ylabel="postsynaptic\nconductance\n(nS)")

    #
    # Synaptic variables
    # Retrieves indexes of spikes in the synaptic monitor using the fact that we
    # are sampling spikes and synaptic variables by the same dt
    #
    logger.info("Synaptic variables")
    ax_syn_var_u = ax_syn_var  # .twinx()
    alpha = 0.4
    for s_i, synapse_mon in enumerate(synapse_monitors):
        s_name = synapse_mon.source.name

        if s_i > len(nrn_idx):
            s_i %= 2

        spike_times = sp_all.t[sp_all.i == nrn_idx[s_i]]  # where index is target
        var_idx = s_i % 2

        s_label = "${}$".format(s_name.replace("_", "").replace("C", "C_{", 1) + "}")
        logger.debug("\t {}".format(s_name))
        spk_index = np.in1d(synapse_mon.t, spike_times)

        # ax_syn_var.plot(synapse_mon.t[spk_index] / time_unit, synapse_mon.x_S[var_idx][spk_index],
        #                 'v', ms=2, color=COLOR.CONN_dict[s_name], alpha=alpha, label=None,
        #                 rasterized=config.RASTERIZED)
        # ax_syn_var_u.plot(synapse_mon.t[spk_index] / time_unit, synapse_mon.u_S[var_idx][spk_index],
        #                   '.', ms=2, color=COLOR.CONN_dict[s_name], alpha=alpha, label=None,
        #                   rasterized=config.RASTERIZED)
        # Super-impose reconstructed solutions
        t = synapse_mon.t  # t vector
        t_spk = Quantity(synapse_mon.t, copy=True)  # Spike times
        for ts in spike_times:
            t_spk[t >= ts] = ts
        ax_syn_var.plot(
            synapse_mon.t / time_unit,
            1 + (synapse_mon.x_S[var_idx] - 1) * np.exp(-(t - t_spk) / tau_d),
            linestyle="-",
            lw=0.8,
            color=COLOR.CONN_dict[s_name],
            alpha=alpha,
            label=s_label,
            rasterized=settings.RASTERIZED,
        )
        ax_syn_var_u.plot(
            synapse_mon.t / time_unit,
            synapse_mon.u_S[var_idx] * np.exp(-(t - t_spk) / tau_f),
            linestyle="--",
            lw=0.5,
            color=COLOR.CONN_dict[s_name],
            alpha=alpha,
            label=None,
            rasterized=settings.RASTERIZED,
        )

        nspikes = np.sum(spk_index)
        x_S_spike = synapse_mon.x_S[var_idx][spk_index]
        u_S_spike = synapse_mon.u_S[var_idx][spk_index]

        ax_w.vlines(
            synapse_mon.t[spk_index] / time_unit,
            np.zeros(nspikes),
            x_S_spike * u_S_spike / (1 - u_S_spike),
            lw=0.5,
            color=COLOR.CONN_dict[s_name],
            rasterized=settings.RASTERIZED,
        )
        ax_w.plot(
            synapse_mon.t[spk_index] / time_unit,
            synapse_mon.dI_S[var_idx][spk_index],
            "-",
            lw=0.5,
            color=COLOR.CONN_dict[s_name],
            rasterized=settings.RASTERIZED,
        )
    ax_syn_var.legend()
    sns.despine(ax=ax_syn_var, top=True, right=False, left=True, bottom=True)

    if ax_syn_var == ax_syn_var_u:
        ax_syn_var.set(ylim=(-0.05, 1.05), ylabel="$x_S$ (-x-) $u_S$ (--•--)")
    else:
        ax_syn_var.set(ylim=(-0.05, 1.05), ylabel="$x_S$ (x)")
        ax_syn_var_u.set(ylim=(-0.05, 1.05), ylabel="$u_S$ (•)")
    ax_syn_var.legend(
        # bbox_to_anchor=(1, 0), loc=3
    )
    ax_w.set(ylabel="W")

    # Adjust axis
    for _ax in ax[:-1]:
        sns.despine(ax=_ax, top=True, right=True, left=False, bottom=True)
        _ax.set_xlabel("")
    sns.despine(ax=ax[-1], top=True, right=True, left=False, bottom=False)
    ax[-1].set_xlabel("Time (s)")
    align_axes(ax)
    return fig, ax
