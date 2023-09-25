# ----------------------------------------------------------------------------------------------------------------------
# VARIABLE NAMES (specifically for rendering in plots)
# ----------------------------------------------------------------------------------------------------------------------

from style.text_math import math_clean

# IONS
CL = cl = "$Cl\it{^-} $"
CLI = CL_i = cli = "$[{}]\mathregular{{_i}}$".format(math_clean(CL))
MILLIMOLAR = mM = "mM"
# CHLORIDE
STATIC_CHLORIDE_STR_ABBR = "Static {}".format(cl)
STATIC_CHLORIDE_STR_LONG = "Static Chloride"
DYNAMIC_CHLORIDE_STR_ABBR = "Dynamic {}".format(cl)
DYNAMIC_CHLORIDE_STR_LONG = "Dynamic Chloride"
ECL0 = f"$E{math_clean(CL)}_0$"
ECL = f"$E{math_clean(CL)}$"

TAU_KCC2 = "$\it{\\tau}_{\\rm{KCC2}}$"

# SYNAPSES
GABA = "GABA"
EGABA = "EGABA"
E_GABA = "$E_{GABA}$"
I_GABA = "$I_{GABA}$"
G_GABA = "$g_{GABA_{max}}$"
G_AMPA = "$g_{AMPA_{max}}$"
G_NMDA = "$g_{NMDA_{max}}$"
GABAA = GABAa = "$GABA_{A}$"
GABAAR = GABAaR = GABAA + "R"
DELTA = "$\Delta $"
NABLA = "$\\nabla $"
DELTAEGABA = f"{DELTA}EGABA"
NABLAEGABA = GRADEGABA = f"{NABLA}EGABA"

# POPULATIONS
POPULATION_MAP = {
    "E": "PC",
    "I": "IN",
    "A": "Average",
}
POPULATION_RATE_MAP = {
    "r_E": "PC",
    "r_I": "IN",
    "r_all": "$\overline{x}$",
}
# population-specific tau
TAU_KCC2_E = TAU_KCC2.replace("KCC2", f"KCC2_{{{POPULATION_MAP['E']}}}")
TAU_KCC2_I = TAU_KCC2.replace("KCC2", f"KCC2_{{{POPULATION_MAP['I']}}}")

CONNECTION_MAP = {
    "C_E_E": "E→E",
    "synapse_mon_cee": "E→E",
    "C_I_E": "E→I",
    "synapse_mon_cie": "E→I",
    "C_E_I": "I→E",
    "synapse_mon_cei": "I→E",
    "C_I_I": "I→I",
    "synapse_mon_cii": "I→I",
}

VESICLES_SYM = "$x_S$"
VESICLES_TEXT = "vesicle pool"
VESICLES_LONG = f"{VESICLES_TEXT} \n [{VESICLES_SYM}]"
EFFICACY_SYM = "$u_S$"
EFFICACY_TEXT = "synaptic efficacy"
EFFICACY_LONG = f"{EFFICACY_TEXT} \n [{EFFICACY_SYM}]"
WEIGHT_SYM = "$w$"
WEIGHT_TEXT = "resources used"
WEIGHT_LONG = f"{WEIGHT_TEXT} \n [{WEIGHT_SYM}]"
# Membrane Potential
MEMBRANE_POTENTIAL = "Membrane Potential"
MILLIVOLTS = "mV"
VOLTAGE_SYMBOL = "$Vm$"
RELATIVE_VOLTAGE_SYMBOL = V = "$V$"
# Distance
MICROMETERS = "$\mu m$"
DISTANCE = "Distance"

# Time
TIME = "Time"
MILLISECONDS = "ms"
SECONDS = "s"
