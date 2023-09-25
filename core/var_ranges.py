import numpy as np


_fp = np.log2(15) % 1  # just get the fractional part
# 2**3.9069 == 15
# 2**7.9069 == 240
# 2**8.9069 == 480
TAU_KCC2_LIST = [
    int(g) for g in np.logspace(start=3 + _fp, stop=7 + _fp, base=2, num=9)
]
# TAU_KCC2_LIST = [int(g) for g in np.logspace(start=3+_fp, stop=8+_fp, base=2, num=11)]
G_GABA_LIST = [25, 50, 100, 200]
