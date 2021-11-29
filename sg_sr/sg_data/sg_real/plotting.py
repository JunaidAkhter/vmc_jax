import numpy as onp 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import numpy as np



E_0_array = np.loadtxt('rlmpo_12_avg_d1_66_gs_try')
print(E_0_array)

#Taking mean along the rng axis. 
E_avg = np.mean(E_0_array, 1)
sigma = np.std(E_0_array, 1)

iterations = np.arange(3)

plt.figure(figsize = (6, 4))
plt.plot(iterations, E_avg,'r', label = r'$MPO_{\mathrm{R}}$, D = 7', linewidth = 2)
#plt.yscale("log")
plt.xlabel("#Iteration")
plt.ylabel(r'$\Delta E/N$')
plt.legend()
#plt.show()
plt.savefig("Rl_comp_62_vs26_2log.pdf")