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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm


from atomic_system import *

import poincare_sphere as ps


## Calcium
calcium = AtomicSys("calcium_levels.txt","calcium_oscillator_strengths.txt",subset=80)


g_state = calcium.state("4s21S0")
e_state = calcium.state("3d4s1D2",m=0)

num_omega = 100

omega_list = np.linspace(0,2.*pi*1000*1000,num=num_omega) #2*pi*GHz


mag_pol = 24.9 #specific polarization angle in degrees


def Q(theta):
    """Q: Electric field polarization parameters -1 <= Q <= 1 """
    theta_rad = theta*(pi/180.0)
    return 1-2*(np.sin(theta_rad))**2

def Theta(Q): # angle in degrees from Q parameter
    sintheta = np.sqrt((1.0-Q)/2.0)
    return np.arcsin(sintheta)*(180.0/pi)

Q_magic = Q(mag_pol)

g_alpha_0 = []
e_alpha_0=[]

g_alpha_mag = []
e_alpha_mag=[]

for j in range(num_omega):
    g_alpha_mag.append(calcium.calculateAlpha(g_state,omega_list[j],Q_magic))
    e_alpha_mag.append(calcium.calculateAlpha(e_state,omega_list[j],Q_magic))
    g_alpha_0.append(calcium.calculateAlpha(g_state,omega_list[j]))
    e_alpha_0.append(calcium.calculateAlpha(e_state,omega_list[j]))

    if(j%20==0):
        print("Completed %i steps"%j)

#np.savetxt('dynamic_polarizabilities_calculated.txt',np.stack([g_alpha_0,e_alpha_0,g_alpha_mag,e_alpha_mag]))



## Calcium Plot
f_list = 1e-3*omega_list/(2.*pi)

nu_list = 1e-3*c/f_list

fig = plt.figure(figsize=(6,4.5))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212,sharex=ax1)


nu_mask = nu_list<1200

ax1.axvline(x=915,color='grey',linestyle='dotted')
ax2.axvline(x=915,color='grey',linestyle='dotted')


ax1.plot(nu_list[nu_mask],np.array(g_alpha_0)[nu_mask],label=("4s$^2$ $^1S_0$ m=0"),color='C0')
ax1.plot(nu_list[nu_mask],np.array(e_alpha_0)[nu_mask],label=("4s3d $^1D_2$ m=0"),color='C1',linestyle='dashed')

ax2.plot(nu_list[nu_mask],np.array(g_alpha_mag)[nu_mask],label=("4s$^2$ $^1S_0$ m=0"),color='C0')
ax2.plot(nu_list[nu_mask],np.array(e_alpha_mag)[nu_mask],label=("4s3d $^1D_2$ m=0"),color='C1',linestyle='dashed')


for ax in [ax1,ax2]:
    ax.grid(visible=True, which='major', color='k', linestyle='-',alpha=0.2)
    ax.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.05)
    ax.minorticks_on()
    ax.set_ylim(-1000,1000)
    ax.set_xlim(300,1200)


plt.xlabel('Wavelength (nm)')

ax1.set_ylabel('$\\alpha(\\theta=0^{\circ})$ (a.u.)')
ax2.set_ylabel('$\\alpha(\\theta=32.5^{\circ})$ (a.u.)')


plt.legend()
plt.tight_layout()
plt.show()

### calculate some specific values

g_state = calcium.state("4s21S0")
e_state = calcium.state("3d4s1D2",m=0)
p_state = calcium.state("4s4p3P0",m=0)
p1_state = calcium.state("4s4p3P1",m=1)


om_915 = e_state.energy/2 #915 nm
om_dc = 0.0000001 #DC electric field


calcium.calculateAlpha(g_state,om_dc,prnt=True)
calcium.calculateAlpha(p_state,om_dc,prnt=True)
calcium.calculateAlpha(e_state,om_dc,prnt=True)

calcium.calculateAlpha(g_state,om_915,prnt=True)
calcium.calculateAlpha(p_state,om_915,prnt=True)
calcium.calculateAlpha(e_state,om_915,prnt=True)


## at 915 nm as a function of polarization angle


num_Q = 50

Q_list = np.linspace(-1.0,1.0,num=num_Q)

e_alpha= np.zeros([5,num_Q]) # 5 for m=-2,-1,0,+1,+2

g_alpha = calcium.calculateAlpha(g_state,om_915,1.0,0.,prnt=True)


for m in [-2,-1,0,1,2]:
    curr_state = calcium.state("3d4s1D2",m=m)
    for j in range(num_Q):
        e_alpha[m,j] = calcium.calculateAlpha(curr_state,om_915,Q_list[j],phi=0.0,prnt=True)


def Theta(Q):
    sintheta = np.sqrt((1.0-Q)/2.0)
    return np.arcsin(sintheta)*(180.0/pi)

theta_list = Theta(Q_list)

# note:
# theta = 0 denotes x polarized light
# Q=-1 denotes x polarized light



## Plot polarizabilities at 915 nm  as function of polarization angle


fig = plt.figure(figsize=(4,3))

plt.plot(theta_list,num_Q*[g_alpha],label="4s$^2$ $^1S_0$ m=0")

for m in [-2,-1,0,1,2]:
    plt.plot(theta_list,e_alpha[m],label="4s3d $^1D_2$ m=%i"%m)

plt.xticks([0,22.5,45,67.5,90])

plt.grid(visible=True, which='major', color='k', linestyle='-',alpha=0.3)
plt.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.1)
plt.minorticks_on()

plt.xlim(0,90)
plt.ylim(100,500)

plt.axvline(mag_pol,color='k',alpha=0.6)

plt.plot(mag_pol,194,'k',marker='o',markersize=10,alpha=0.3)

plt.xlabel("Field polarization $\\theta$ (deg.)")
plt.ylabel("Polarizability $\\alpha$(915 nm) (a.u.)")

plt.legend(fontsize='x-small')
plt.tight_layout()

plt.show()


## convert to light shift and calculate slope around magic polarization

def LightShift(polarizability):
    """returns light shift in units of 2*pi*Hz/(W/m^2)
    # polarizability in atomic units (kC*a0^3) """
    return (-Z0*polarizability*alpha_au)/hbar


def Alpha(light_shift):
    """returns polarizability in atomic units of (kC*a0^3)
    # light shift in units of 2*pi*Hz/(W/m^2)"""
    return -(hbar*light_shift/Z0)/alpha_au


diff_alpha = e_alpha[0] - g_alpha #e_alpha[0] is mJ=0

light_shift_SI = LightShift(diff_alpha) #2*pi*Hz/(W/m^2)

light_shift = 1e4*light_shift_SI/(2*pi) #Hz/(W/cm^2)


theta_near_mag = theta_list[np.abs(theta_list-mag_pol)<3]
shift_near_mag = light_shift[np.abs(theta_list-mag_pol)<3]

def fit_func(x,m,b):
    return m*x+b

from scipy.optimize import curve_fit
popt,pcov = curve_fit(fit_func,theta_near_mag,shift_near_mag)


fig = plt.figure(figsize=(4,3))

ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.plot(theta_list,light_shift)
ax1.plot(theta_list[theta_list<50],fit_func(theta_list[theta_list<50],*popt),'--')
ax1.axvline(mag_pol,color='k',alpha=0.6,linestyle='dotted')
ax1.axhline(0,color='k',linewidth=0.5)


ax1.set_xlabel("Polarization $\\theta$ (deg.)")
ax1.set_ylabel("Light shift (Hz/W/cm$^2$)")

ax2.set_ylabel("Differential polarizability $\Delta\\alpha$ (a.u.)")
ylim = np.array([-20,5])

ax1.set_xlim(0,90)
ax1.set_ylim(*ylim)
ax2.set_ylim(*(Alpha(1e-4*2*pi*ylim)))

ax1.set_xticks([0,22.5,45,67.5,90])

ax1.grid(visible=True, which='major', color='k', linestyle='-',alpha=0.2)
ax1.grid(visible=True, which='minor', color='k', linestyle='-', alpha=0.05)
ax1.minorticks_on()


plt.tight_layout()
plt.show()

## Plot on Poincare Sphere
e_alpha_m0 = e_alpha[0]

n = len(e_alpha_m0)

diff_alpha = e_alpha_m0 - g_alpha


def Stokes(theta,phix):
    Ex = np.sin(theta)
    Ez = np.cos(theta)
    S1 = Ez**2 - Ex**2
    S2 = 2.*Ex*Ez*np.cos(phix)
    S3 = 2.*Ex*Ez*np.sin(phix)
    return S1,S2,S3

phix_list = np.linspace(0,2.*pi,num=n) #phase difference between z^ and x^ components of electric field

# abridged to make the figure render faster, don't need the whole back of the sphere'
#phix_list = np.linspace(pi/2,3*pi/2,num=n) #phase diff between z^ and x^ components of electric field


theta_mesh,phix_mesh = np.meshgrid(3.14*theta_list/180,phix_list)
s1,s2,s3 = Stokes(theta_mesh,phix_mesh)

# no phi dependence so just copy the array 50 times
diff_alpha_T = np.array([diff_alpha.T for i in range(n)])


fig = plt.figure()
ax = plt.axes(projection='3d',computed_zorder=False)


ax.set_box_aspect((1,1,1))  # aspect ratio is 1:1:1

p=ps.Poincare(fig=fig,axes=ax)
p.frame_color='black'
p.frame_alpha = 0.0
p.axes_alpha=0.1
p.sphere_alpha = 0.0

p.set_label_convention("polarization stokes")
p.ylabel = ['$\\updownarrow\\hat{z}\\;\\;$', '$\\;\\;\\leftrightarrow\\hat{x}$']


norm_plt = mcolors.Normalize(vmin=-200, vmax=200)
m = cm.ScalarMappable(cmap=plt.cm.seismic_r, norm=norm_plt)
m.set_array([])
facecolor_vals = np.array(norm_plt(diff_alpha_T),dtype=np.float64)

# surface plot
surf = ax.plot_surface(s1,s2,s3,rcount=n,ccount=n, facecolors=plt.cm.seismic_r(facecolor_vals),alpha=0.8,linewidth=0, antialiased=True,zorder=10)

ax.set_axis_off()

# Add a color bar which maps values to colors.
cbar=fig.colorbar(m,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=12)
#ax.view_init(elev=20., azim=115)

p.render(fig=fig,axes=ax)
plt.show()

