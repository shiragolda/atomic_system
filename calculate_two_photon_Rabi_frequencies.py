import numpy as np

from scipy.constants import pi, hbar, c, e, m_e, epsilon_0, Boltzmann, alpha, physical_constants
kC = 1/(4*pi*epsilon_0)
a0 = physical_constants["atomic unit of length"][0]
Z0 = physical_constants["characteristic impedance of vacuum"][0]
kB = Boltzmann
alpha_au = 4*pi*epsilon_0*a0**3        # 1 a.u. * (1 V/cm)**2 /h in Hz
E_h = alpha**2 * m_e * c**2  # Hartree - unit of energy in au
omega_au = E_h/hbar  # au frequency
mu_B = physical_constants["Bohr magneton"][0]


from atomic_system import *

import poincare_sphere as ps

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.collections import LineCollection


## Calcium
calcium = AtomicSys("calcium_levels.txt","calcium_oscillator_strengths.txt",subset=50)

g_state = calcium.state("4s21S0")
e_state = calcium.state("3d4s1D2",m=0)


omega_ca =  e_state.energy/2 #clock frequency   
# omega is the dressing frequency in 2*pi*GHz




## sanity checks

ca_4s4p1p1 = calcium.state('4s4p1P1',m=0)
ca_d_odd_0 = calcium.state('3d4p1D2',m=0)
ca_d_odd_1 = calcium.state('3d4p1D2',m=1)


print(calcium.E1Moment(ca_4s4p1p1,g_state))
print(calcium.E1Moment(ca_d_odd_0,e_state))
print(calcium.E1Moment(ca_d_odd_1,e_state,q=1))

## Rabi frequencies

num_Q = 20
num_phix = 3

Q_list = np.linspace(-1.0,1.0,num=num_Q) 
phix_list = np.linspace(0,pi/2.0,num=num_phix) #phase diff between z^ and x^ components of electric field


def Theta(Q):
    sintheta = np.sqrt((1.0-Q)/2.0)
    return np.arcsin(sintheta)*(180.0/pi)

theta_list = Theta(Q_list)

rabi_ca = np.zeros((num_phix,num_Q))

for i in range(num_phix):
    for j in range(num_Q):
        rabi_ca[i][j] = abs(complex(calcium.twoPhotonE1Moment(g_state,e_state,Q=Q_list[j],phi=phix_list[i])))
    print("Finished phi %i"%i)

#rabi_ca has units of (e*a0)^2/E_h 

##Plot transition probabilities 

mag_pol = 24.9


rabi0 = max(rabi_ca[0,:])

rabi_normalized = [s**2/rabi0**2 for s in rabi_ca]

rabi = [s**2 for s in rabi_ca]

fig = plt.figure(figsize=(4,3))

lines = ["-","--","-.",":"]
colors = ['C0','C1','C2']

ax1 = plt.subplot(111)

ax1.axvline(mag_pol,color='grey',linestyle='dotted')

for i in range(num_phix):
    ax1.plot(theta_list,rabi_normalized[i],linewidth=2,label='$\phi$ = %.0f$^{\circ}$'%float(phix_list[i]*(180.0/pi)),linestyle=lines[i],color=colors[i])


xlabel = 'Linear polarization angle, $\\theta$'
ylabel = 'Normalized Rabi frequency $|\Omega|^2$'


ax1.set_xlabel(xlabel,fontsize=10)
ax1.set_ylabel(ylabel,fontsize=10)
ax1.set_xticks(np.linspace(0,90,num=5))
ax1.legend()

ax1.set_xlim(0,90)
ax1.set_ylim(0,1)

ax1.grid(visible=True, which='major', color='k', linestyle='-',alpha=0.2)
ax1.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.05)
ax1.minorticks_on()

plt.tight_layout()

plt.show()

# Q=-1 denotes x polarized light -> theta = 90
# Q=1 denotes z polarizaed light -> theta = 0

