from __future__ import absolute_import, division, print_function

import json
import time
from collections import OrderedDict
from copy import deepcopy
from os import makedirs, path

import brian2.numpy_ as np
import matplotlib
from brian2 import (
    Network,
    NeuronGroup,
    PoissonInput,
    Synapses,
    collect,
    defaultclock,
    magic_network,
    prefs,
    run,
    start_scope,
)
from brian2.devices import all_devices, device, get_device, seed, set_device
from brian2.monitors import PopulationRateMonitor, SpikeMonitor, StateMonitor
from brian2.units import Hz, mM, ms, mV, nF, nS, second, um
from brian2.units.constants import faraday_constant as F
from brian2.units.constants import gas_constant as R
from brian2.units.constants import kelvin, zero_celsius
from matplotlib import pyplot as plt
from matplotlib.cbook import flatten

from settings import logging

logger = logging.getLogger("sim")
# logger.addHandler(handler)

################################################################################
# Settings
################################################################################
man_connections = False  # number of connections (from w=0 to w=1) changes
man_weights = False  # weight of connections changes (from w=0 to w>1)

# Brian 2 settings
device_directory = ".cpp"
device_debug = logger.level == "DEBUG"
if path.exists("C:\\Program Files (x86)"):
    prefs.codegen.cpp.msvc_vars_location = (
        "C:\\Program Files (x86)\\Microsoft Visual "
        "Studio\\2017\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat"
    )
if not (man_connections or man_weights):
    set_device(
        "cpp_standalone",
        directory=device_directory,
        build_on_run=False,
        debug=device_debug,
    )
celsius_temp = 37  # Temperature
T = celsius_temp * kelvin + zero_celsius
rt_f = R * T / F

__default_monitors = {
    "sp_all": True,
    "r_E": True,
    "r_I": True,
    "r_all": True,
    "Mg2": True,
    "state_mon": ["v", "E_GABA", "I_GABA_rec", "E_Cl", "g_NMDA", "g_AMPA", "g_GABA"],
    "synapse_mon": ["x_S", "u_S", "dI_S"],
}


def clean_device():
    device.build(directory=device_directory, clean=True)


def nernst(C_out, C_in, charge):
    """Nernst equation:
    E = R * T / (F z) * ln([X]o / [X]i)
    """
    return rt_f * (1 / charge) * np.log(C_out / C_in)


def get_sim_name(**kwargs):
    arg_vars = OrderedDict(sorted(locals()["kwargs"].items(), key=lambda _x: _x[0]))
    keys = list(arg_vars.keys())
    for arg_key in keys:
        if arg_key.startswith("__") or arg_key == "key":
            del arg_vars[arg_key]
        elif arg_vars[arg_key] is None or callable(arg_vars[arg_key]):
            del arg_vars[arg_key]
    sim_name = json.dumps(arg_vars).replace('"', "").replace(": ", "=")[1:-1]
    return sim_name


def single_run(
    N=None,
    duration=None,
    dt=None,  # Simulation params
    Mg2_t0=None,
    zero_mag_wash_rate=None,  # Seizure params
    benzo_onset_t=None,
    benzo_wash_rate=None,
    benzo_strength=None,
    benzo_off_t=None,  # drug params
    p=None,
    p_ee=None,
    p_ei=None,
    p_ie=None,
    p_ii=None,  # Connection params
    w: str = None,
    w_ee: str = None,
    w_ei: str = None,
    w_ie: str = None,
    w_ii: str = None,  # weight params
    U_0=None,
    tau_d=None,
    tau_f=None,  # STP params
    g_AMPA_max=None,
    g_NMDA_max=None,
    g_GABA_max=None,  # conductances
    E_Cl_0=None,
    E_Cl_target=None,
    E_Cl_end=None,
    E_Cl_pop=None,  # ECl params
    length=None,  # neuron params
    dyn_cl=None,
    manual_cl=None,  # dynamic/manual arg
    tau_KCC2_E=None,
    tau_KCC2_I=None,
    num_ecl_steps=None,  # dynamic + manual params
    __build=True,
    __save_run=False,
    __monitors=None,
    nrn_idx_i=None,  # Brian2 args
    run_seed=None,
    __plot=False,
    __device_directory=None,
):
    sim_name = get_sim_name(**locals())
    if __monitors is None:
        __monitors = deepcopy(__default_monitors)
    else:
        __monitors = {**__default_monitors, **__monitors}  # overwrite defaults
    start_time = time.time()
    sum_time = 0
    __facilitation = True  # disable by setting tau_f to 0

    logger.info(sim_name)
    file_name = path.join("temp", sim_name + ".npz")
    if path.exists(file_name):
        set_device("runtime")
    else:
        device_directory = __device_directory or ".cpp"
        set_device(
            "cpp_standalone",
            directory=device_directory,
            build_on_run=False,
            debug=device_debug,
        )

    if get_device() == all_devices["cpp_standalone"]:
        device.reinit()
        device.activate(
            directory=device_directory, build_on_run=False, debug=device_debug
        )
    seed(run_seed or 11000)  # to get identical figures for repeated runs
    start_scope()

    ################################################################################
    # Settings
    ################################################################################
    dyn_cl = dyn_cl or False
    # choose type of manual cl change (specific to a certain population)
    manual_cl = {False: 0, True: 1, None: 1, "both": 1, "E": 2, "I": 3}[manual_cl]
    E_Cl_pop = {None: 1, "both": 1, "E": 2, "I": 3}[E_Cl_pop]
    manual_cl = not dyn_cl and (manual_cl or True)

    ################################################################################
    # Model parameters
    ################################################################################
    # General parameters
    duration = (duration or 90) * second  # Total simulation time

    # Seizure parameters
    zero_mag_onset_t = 0.0 * second
    zero_mag_wash_rate = (zero_mag_wash_rate or 5) * second
    zero_mag_off_t = duration  # - zero_mag_wash_rate

    benzo_onset_t = False if benzo_onset_t is None else benzo_onset_t * second
    benzo_wash_rate = (benzo_wash_rate or 5) * second
    benzo_strength = 1.0 if benzo_strength is None else benzo_strength
    benzo_off_t = (benzo_off_t or duration / second) * second

    # Network
    N = N or 1000  # Number of total neurons
    prop_I = 0.2  # Proportion of neurons that are inhibitory
    N_E = int(N * (1 - prop_I))  # Number of excitatory neurons
    N_I = int(N * prop_I)  # Number of inhibitory neurons
    assert N_E + N_I == N

    # Neuron
    V_thr = -50.0 * mV  # Firing threshold
    V_reset = -65.0 * mV  # Reset potential
    C_m_E = 0.550 * nF  # Membrane capacitance (excitatory population)
    C_m_I = 0.450 * nF  # Membrane capacitance (inhibitory population)
    tau_r_E = 2.0 * ms  # Refractory period (excitatory population)
    tau_r_I = 1.0 * ms  # Refractory period (inhibitory population)

    # Size
    L = (length or 7.5) * um  # Length
    a = L / 2  # Radius
    volume_E = 4.0 / 3.0 * np.pi * (L / 2) * (a**2)  # volume oblate spheroid
    volume_I = volume_E * 2 / 3

    # Leak
    E_leak = -70.0 * mV  # Leak reversal potential
    g_l_E = 20.0 * nS  # Leak conductance (excitatory population)
    g_l_I = 20.0 * nS  # Leak conductance (inhibitory population)

    # Synapse parameters
    p = p or 0.02  # Connection probability
    p_E_E = p_ee or p  # E from E
    p_I_E = p_ie or p  # I from E
    p_E_I = p_ei or (p * 2)  # E from I (default is double)
    p_I_I = p_ii or (p * 2)  # I from I (default is double)

    # Synaptic plasticity params
    U_0 = U_0 or 0.01  # Synaptic release probability at rest
    tau_d = (tau_d or 10.0) * second  # Synaptic depression rate
    if tau_f == 0:
        # not facilitating
        __facilitation = False
        __monitors["synapse_mon"].remove("u_S")
    else:
        tau_f = (tau_f or 0.5) * second  # Synaptic facilitation rate

    # Scale synaptic weights
    J0 = 1000
    scale = J0 / N
    logger.debug("scale = {:.4f}".format(scale))

    # AMPA (excitatory)
    g_AMPA_max = (g_AMPA_max or 5.0) * nS * scale
    g_AMPA_max_ext = 2.0 * nS
    tau_AMPA = 2.0 * ms
    E_AMPA = 0.0 * mV

    # NMDA (excitatory)
    g_NMDA_max = (g_NMDA_max or 5.0) * nS * scale
    tau_NMDA_rise = 2.0 * ms
    tau_NMDA_decay = 100.0 * ms
    alpha = 0.5 / ms
    Mg2_t0 = Mg2_t0 or 1.0
    E_NMDA = 0.0 * mV

    # GABAergic (inhibitory)
    if g_GABA_max is None:
        g_GABA_max = 50.0 * nS * scale
    else:
        g_GABA_max = g_GABA_max * nS * scale
    tau_GABA = 10.0 * ms
    E_Cl_0 = (E_Cl_0 or -88.0) * mV  # Reversal potential for Chloride (starting value)
    E_Cl_target = (E_Cl_target or -88.0) * mV  # Target ECl for KCC2
    E_HCO3 = -18 * mV  # Reversal potential for Bicarbonate
    pcl = 0.8  # Permeability proportion of Chloride through GABAA
    phco3 = 1 - pcl  # Permeability proportion of Bicarbonate through GABAA
    C_Cl_out = 135 * mM  # External chloride concentration (constant)
    C_Cl_in = C_Cl_out * np.exp(
        E_Cl_0 / rt_f
    )  # Internal chloride concentration (indirectly varied through d_ECl/dt)
    ecl_alpha_E = rt_f / (F * volume_E * C_Cl_out)
    ecl_alpha_I = rt_f / (F * volume_I * C_Cl_out)
    ecl_beta = 1 / rt_f
    E_GABA_0 = (
        pcl * E_Cl_0 + phco3 * E_HCO3
    )  # Reversal potential for GABAA (Ohmic formulation using Cl and HCO3
    #  permeabilities)
    logger.debug("EGABA = {:.2f}".format(E_GABA_0 / mV))
    tau_KCC2_E = (tau_KCC2_E or 60) * second
    tau_KCC2_I = (tau_KCC2_I or 60) * second

    # External stimuli
    rate_ext = 2 * Hz
    C_ext = 800
    
    # Chloride dynamics
    E_Cl_end = (E_Cl_end or -60) * mV  # Final ECl to reach when stepping ECl
    num_ecl_steps = num_ecl_steps or 1  # Number of changes to ECl when manual cl.
    static_cl_dt = zero_mag_onset_t + (zero_mag_off_t - zero_mag_onset_t) / (
        num_ecl_steps + 1
    )
    static_cl_units = -1 * (E_Cl_0 - E_Cl_end) / (num_ecl_steps + 1)

    logger.debug("static_cl_units = {:.4f}mV".format(static_cl_units / mV))

    # Connection dynamics
    static_conn_dt = 5000 * ms
    static_weights_step = 1
    p_O_ = 0.1  # Connection probability by end
    p_conn_values = np.linspace(0, p_O_, 1 + int(duration / static_conn_dt / 2))
    if manual_cl:
        p_conn_values = np.append(p_conn_values, p_conn_values)
    S_change = {"C_E_E": None}

    ################################################################################
    # Model definition
    ################################################################################
    neuron_eqs = """
        # variables that differ between excitatory and inhibitory populations
        g_l: siemens
        C_m: farad
        tau_r: second
        Mg2: 1

        dv / dt = (- g_l * (v - E_leak) - I_syn) / C_m : volt

        I_syn = I_AMPA_ext + I_AMPA_rec + I_NMDA_rec + I_GABA_rec : amp

        I_AMPA_ext = g_AMPA_ext * (v - E_AMPA) : amp
        I_AMPA_rec = g_AMPA * (v - E_AMPA) : amp
        dg_AMPA_ext / dt = - g_AMPA_ext / tau_AMPA : siemens
        dg_AMPA / dt = - g_AMPA / tau_AMPA : siemens

        I_NMDA_rec = g_NMDA * (v - E_NMDA) / (1 + Mg2 * exp(-0.062 * v / mV) / 3.57) : amp
        g_NMDA = g_NMDA_max * g_NMDA_tot : siemens
        g_NMDA_tot : 1  # from synapse model equations

        E_GABA = pcl * E_Cl + phco3 * E_HCO3 : volt
        I_GABA_rec = g_GABA * (v - E_GABA) : amp
        dg_GABA / dt = - g_GABA / tau_GABA : siemens
    """

    # Chloride equations depending on settings
    if dyn_cl:
        logger.debug("dyn_cl")
        # Dynamic chloride model equations
        _dE_Cl_influx = "ecl_alpha * exp( -(ecl_beta * E_Cl)) * g_GABA * pcl * (v-E_Cl)"
        _dE_Cl_efflux = "(E_Cl - E_Cl_target)/tau_KCC2"
        dE_Cl_eqn = f"""
            tau_KCC2: second
            ecl_alpha : metre**2 * kilogram * second**-4 * amp**-2
            dE_Cl / dt = {_dE_Cl_influx} - {_dE_Cl_efflux} : volt
            """
        neuron_eqs += dE_Cl_eqn
    elif manual_cl:
        logger.debug("manual_cl")
        # ECl incremented in units
        neuron_eqs += """
            E_Cl = E_Cl_0 + static_cl_units * increment_ecl_index: volt
            increment_ecl_index: 1
        """
    else:
        neuron_eqs += """E_Cl: volt"""

    # Create initial master population
    # noinspection PyTypeChecker
    neurons = NeuronGroup(
        N,
        neuron_eqs,
        threshold="v > V_thr",
        reset="v = V_reset",
        refractory="tau_r",
        method="euler",
    )
    # Separate into 2 (E & I) populations
    P_E = neurons[:N_E]
    P_I = neurons[N_E:]
    P_E.g_l = g_l_E
    P_E.C_m = C_m_E
    P_E.tau_r = tau_r_E
    P_I.g_l = g_l_I
    P_I.C_m = C_m_I
    P_I.tau_r = tau_r_I
    if dyn_cl:
        # time constants for chloride extrusion are different
        P_E.tau_KCC2 = tau_KCC2_E
        P_I.tau_KCC2 = tau_KCC2_I
        # volumes are different
        P_E.ecl_alpha = ecl_alpha_E
        P_I.ecl_alpha = ecl_alpha_I

    # Synapses equations
    # from https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
    synapses_eqs = """
        local_g_GABA_max : siemens     # allows change using run_regularly
        w : 1       # weighted connection
        dI_S : 1    # amount of resources used
        # Fraction of synaptic neurotransmitter resources available:
        dx_S/dt = (1 - x_S)/tau_d : 1 (event-driven)
        """
    if __facilitation:
        synapses_eqs += """
        # Utilisation of releasable neurotransmitter per single action potential:
        du_S/dt = -u_S/tau_f     : 1 (event-driven)
        """
        synapses_action = """
            u_S += U_0 * (1 - u_S) # facilitate synapse
            dI_S = u_S * x_S * w
            x_S -= dI_S
        """
        logger.debug("facilitating synapses")
    else:
        synapses_action = """
            dI_S = U_0 * x_S * w
            x_S -= dI_S
        """
        logger.debug("non-facilitating synapses")

    eqs_glut = """
        g_NMDA_tot_post = g_NMDA_syn : 1 (summed)
        dg_NMDA_syn / dt = - g_NMDA_syn / tau_NMDA_decay + alpha * x * (1 - g_NMDA_syn) : 1 (clock-driven)
        dx / dt = - x / tau_NMDA_rise : 1 (clock-driven)
    """

    eqs_pre_glut = f"""{synapses_action}
        g_AMPA += g_AMPA_max * dI_S
        x += 1 * dI_S   # easier units to work with than placing g_NMDA_max here
    """

    eqs_pre_gaba = f"""{synapses_action}
        g_GABA += local_g_GABA_max * dI_S
    """

    # logger.info("SYNAPSE EQUATIONS: \n\t {}".format(synapses_eqs))

    # E to E
    C_E_E = Synapses(
        P_E,
        P_E,
        model=synapses_eqs + eqs_glut,
        on_pre=eqs_pre_glut,
        method="euler",
        name="C_E_E",
    )
    C_E_E.connect(p=p_E_E)
    C_E_E.w = w_ee or w or "1"

    # E to I (I from E)
    C_I_E = Synapses(
        P_E,
        P_I,
        model=synapses_eqs + eqs_glut,
        on_pre=eqs_pre_glut,
        method="euler",
        name="C_I_E",
    )
    C_I_E.connect(p=p_I_E)
    C_I_E.w = w_ie or w or "1"

    # I to E (E from I)
    C_E_I = Synapses(
        P_I, P_E, model=synapses_eqs, on_pre=eqs_pre_gaba, method="euler", name="C_E_I"
    )
    C_E_I.connect(p=p_E_I)
    C_E_I.w = w_ei or w or "1"

    # I to I
    C_I_I = Synapses(
        P_I, P_I, model=synapses_eqs, on_pre=eqs_pre_gaba, method="euler", name="C_I_I"
    )
    C_I_I.connect(p=p_I_I)
    C_I_I.w = w_ii or w or "1"

    # external noise
    C_P_E = PoissonInput(P_E, "g_AMPA_ext", C_ext, rate_ext, "g_AMPA_max_ext")
    C_P_I = PoissonInput(P_I, "g_AMPA_ext", C_ext, rate_ext, "g_AMPA_max_ext")

    S_list = [C_E_E, C_I_E, C_E_I, C_I_I]
    for S in S_list:
        if S.name in S_change:
            S_change[S.name] = S

    # ##############################################################################
    # # Monitors
    # ##############################################################################
    sp_all = SpikeMonitor(neurons, name="sp_all") if __monitors["sp_all"] else None

    mg2_mon = (
        StateMonitor(P_E, "Mg2", record=np.arange(1)) if __monitors["Mg2"] else None
    )

    r_E = PopulationRateMonitor(P_E, name="r_E") if __monitors["r_E"] else None
    r_I = PopulationRateMonitor(P_I, name="r_I") if __monitors["r_I"] else None
    r_all = (
        PopulationRateMonitor(neurons, name="r_all") if __monitors["r_all"] else None
    )

    # We record some additional data from single neurons
    # useful to record from 0 and 1 to distinguish connections (else if neuron 0 spikes, then overlapping traces are
    # produced)
    nrn_idx_i = nrn_idx_i or [0, 1]
    nrn_idx = list(flatten([[n, n + N_E] for n in nrn_idx_i]))
    nrn_idx_type = ["E", "I"] * len(nrn_idx_i)
    # Record conductances and membrane potential of neurons
    state_mon = (
        StateMonitor(neurons, __monitors["state_mon"], record=nrn_idx)
        if __monitors["state_mon"]
        else None
    )
    synapse_monitor_kwargs = dict(
        variables=__monitors["synapse_mon"], record=nrn_idx_i, when="after_synapses"
    )
    synapse_mon_cee = (
        StateMonitor(C_E_E, **synapse_monitor_kwargs)
        if __monitors["synapse_mon"]
        else None
    )
    synapse_mon_cie = (
        StateMonitor(C_I_E, **synapse_monitor_kwargs)
        if __monitors["synapse_mon"]
        else None
    )
    synapse_mon_cei = (
        StateMonitor(C_E_I, **synapse_monitor_kwargs)
        if __monitors["synapse_mon"]
        else None
    )
    synapse_mon_cii = (
        StateMonitor(C_I_I, **synapse_monitor_kwargs)
        if __monitors["synapse_mon"]
        else None
    )
    # synapse_monitors = [synapse_mon_cee, synapse_mon_cie, synapse_mon_cei, synapse_mon_cii]

    # ##############################################################################
    # # Initialisation
    # ##############################################################################
    neurons.Mg2 = "Mg2_t0"
    neurons.v = "E_leak + rand()*(V_thr-E_leak)"
    neurons.g_AMPA = "rand()*g_AMPA_max*U_0"
    neurons.g_NMDA_tot = "rand()*g_NMDA_max*U_0 / nS"
    neurons.g_GABA = "rand()*g_GABA_max*U_0"

    C_E_I.local_g_GABA_max = "g_GABA_max"
    C_I_I.local_g_GABA_max = "g_GABA_max"

    # Add seizure and chloride dynamics
    if manual_cl:
        logger.info(
            "Incrementing E_Cl by {:.2f}mV every {:.2f}s for population: {:>s}".format(
                static_cl_units / mV,
                static_cl_dt / second,
                {1: "neurons", 2: "P_E", 3: "P_I"}[manual_cl],
            )
        )
        # explicitly set ECl at time points for a given population
        population = {1: neurons, 2: P_E, 3: P_I}[manual_cl]
        population.increment_ecl_index = "-1"
        population.run_regularly("""increment_ecl_index += 1""", dt=static_cl_dt)
    else:
        population = {1: neurons, 2: P_E, 3: P_I}[E_Cl_pop]
        population.E_Cl = "E_Cl_0"

    if zero_mag_onset_t >= 0:
        change_per_ms = Mg2_t0 / (zero_mag_wash_rate / ms)
        neurons.run_regularly(
            """
            wash_in = t>=zero_mag_onset_t and t<=zero_mag_off_t
            Mg2 = clip(Mg2 - wash_in*{0} + (1-wash_in)*{0},0,Mg2_t0)""".format(
                change_per_ms
            ),
            dt=1 * ms,
        )
        logger.info(
            "[{}] mM of Mg will be removed (at {}s) and added back (at {}s) over {}s at a rate_ext of [{}]/ms)".format(
                Mg2_t0,
                zero_mag_onset_t,
                zero_mag_off_t,
                zero_mag_wash_rate / second,
                change_per_ms,
            )
        )

    if benzo_onset_t is not False:
        new_g_GABA_max = g_GABA_max * benzo_strength
        change_per_ms = (new_g_GABA_max - g_GABA_max) / (benzo_wash_rate / ms)
        if change_per_ms != 0:
            lower_bound = g_GABA_max if change_per_ms > 0 else new_g_GABA_max
            upper_bound = new_g_GABA_max if change_per_ms > 0 else g_GABA_max
            clip_text = (
                "clip(local_g_GABA_max/nS + "
                f"wash_in*{change_per_ms/nS} - wash_out*{change_per_ms/nS},{lower_bound/nS},{upper_bound/nS})*nS"
            )
            eq = """
                wash_in = t>=benzo_onset_t and t<=benzo_off_t
                wash_out = t > benzo_off_t
                local_g_GABA_max = {clip_text}
            """.format(
                clip_text=clip_text
            )
            C_E_I.run_regularly(eq, dt=1 * ms)
            C_I_I.run_regularly(eq, dt=1 * ms)

            logger.info(
                f"benzo will be washed in over {benzo_wash_rate/second} "
                f"(changing at {change_per_ms/nS} nS per ms), "
                f"starting at {benzo_onset_t} and washed out at {benzo_off_t}"
            )

    for _S in S_list:
        _S.x_S = "clip(U_0*2,0,1)"  # initial resources available

    defaultclock.dt = dt or defaultclock.dt

    if get_device() != all_devices["cpp_standalone"]:
        net = Network(collect())
        # net.add(synapse_monitors)
        file_name = path.join("temp", sim_name + ".npz")
        if path.exists(file_name):
            logger.info("loading")
            states = np.load(file_name, allow_pickle=True)
            states_dict = {_key: states[_key].item() for _key in states}
            restored = False

            while not restored:
                try:
                    net.set_states(states_dict)
                    restored = True
                except KeyError as k:
                    idx1 = k.args[0].index("'") + 1
                    idx2 = k.args[0].index("'", idx1)
                    del states_dict[k.args[0][idx1:idx2]]
            return net, dict(locals())
        net.store(sim_name)

    setup_time = time.time()
    setup_dt = setup_time - start_time
    sum_time += setup_dt
    logger.info("took {:.2f}s to setup".format(setup_dt))

    logger.info("running")
    if get_device() != all_devices["cpp_standalone"]:
        net.restore(sim_name)
    run_time_start = time.time()
    if man_connections or man_weights:
        # change connections
        for idx, p_conn_step in enumerate(p_conn_values):
            if man_weights:
                w_val = idx * static_weights_step
            else:
                w_val = 1
            logger.info("w_val={}".format(w_val))

            for S_name, S in S_change.items():
                if man_connections:
                    logger.info("{}: p_conn_step = {}".format(S_name, p_conn_step))
                    # increase number of connections
                    if p_conn_step == 0:
                        S.w = "0"
                    else:
                        N_pop = N_I if S_name.endswith("_I") else N_E
                        p0_idx = int(N_pop * p_conn_step)
                        S.w[:p0_idx] = w_val
                else:
                    S.w = w_val
                logger.info("{}: # that are >0 = {:.0f}".format(S_name, np.sum(S.w > 0.0)))
                logger.info("{}: # that are >1 = {:.0f}".format(S_name, np.sum(S.w > 1.0)))
            net.run(static_conn_dt, report="text")
    else:
        run(duration, report="text")
    if __build and get_device() == all_devices["cpp_standalone"]:
        try:
            device.build(directory=device_directory, debug=device_debug)
        except RuntimeError as e:
            logger.error(e)
            logger.error("rebuilding with clean=True")
            device.build(directory=device_directory, debug=device_debug, clean=True)

    run_time = time.time()
    run_dt = run_time - run_time_start
    logger.info("took {:.2f}s to run".format(run_dt))
    if __save_run and get_device() != all_devices["cpp_standalone"]:
        states = net.get_states(read_only_variables=False)
        makedirs("temp", exist_ok=True)
        np.savez_compressed(file_name, **states)
    if __plot:
        from style.plot_group import activity_plot, plot_hierarchy, plot_states

        plot_hierarchy(
            N,
            N_E,
            sp_all,
            r_all,
            r_E,
            r_I,
            state_mon,
            V_thr,
            nrn_idx=[0, 1],
            time_unit=second,
        )

        activity_plot(
            N,
            N_E,
            sp_all,
            r_all,
            r_E,
            r_I,
            mg2_mon,
            state_mon,
            zero_mag_onset_t=zero_mag_onset_t,
            zero_mag_off_t=zero_mag_off_t,
            zero_mag_wash_rate=zero_mag_wash_rate,
            time_unit=second,
        )
        plot_states(
            state_mon,
            [synapse_mon_cee, synapse_mon_cie, synapse_mon_cei, synapse_mon_cii],
            V_thr,
            V_reset,
            sp_all,
            tau_d,
            tau_f,
            nrn_idx=[0, 0, 0, 0],
            nrn_idx_type=["E"],
            # nrn_idx=nrn_idx, nrn_idx_type=nrn_idx_type,
            time_unit=second,
        )

    return magic_network, dict(locals())


if __name__ == "__main__":
    net, result = single_run(
        zero_mag_wash_rate=5, E_Cl_end=-40, __plot=True, benzo_onset_t=20
    )
    if matplotlib.get_backend().lower() not in ["agg", "pdf"]:
        plt.show()
