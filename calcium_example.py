## Calcium example
from atomic_system import *

calcium = AtomicSys("calcium_levels.txt","calcium_oscillator_strengths.txt",subset=200) #first 200 energy levels of calcium


g_state = calcium.state("4s21S0")
p_state = calcium.state("4s4p1P1",m=0)
e_state = calcium.state("3d4s1D2",m=0)
r_state = calcium.state("4snp1P1",m=1)
n_state = calcium.state("4s5p1P1",m=-1)


m1_state = calcium.state("4s4p3P0",m=0)
m2_state = calcium.state("4s4p3P1",m=0)

printLevelData(g_state)
printLevelData(r_state)
printLevelData(m2_state)


## calculate dynamic polarizabilities

omega0 =  e_state.energy/2 #clock frequency    omega = 0.0586

calcium.calculateAlpha(g_state,omega=0.0001,Q=1.0,prnt=True)
calcium.calculateAlpha(g_state,omega=omega0,Q=1.0,prnt=True)

print(calcium.E1Moment(g_state,p_state))
print(calcium.M1Moment(m1_state,m2_state))

## calculate two-photon transition Rabi frequency

# two photon clock rabi frequency
print(float(calcium.twoPhotonE1Moment(g_state,e_state)))

Rabi_coeff = ((e*a0)**2/E_h)*calcium.twoPhotonE1Moment(g_state,e_state,Q=1.0)

beam_waist = 0.2e-3 #m
beam_power = 0.5 #W

beam_cross_section_area = np.pi*(beam_waist)**2 #m^2
beam_intensity = 2*beam_power/beam_cross_section_area #W/m^2
electric_field = np.sqrt(2*Z0*beam_intensity) #V/m

Rabi_freq = (Rabi_coeff*(electric_field**2))/(2*pi*hbar) #Hz
print(float(Rabi_freq)) #Hz

## diagonalize and get eigenvalues of Hamiltonian

calcium_subset = AtomicSys("calcium_levels.txt","calcium_oscillator_strengths.txt",subset=10)


H0 = calcium_subset.genFreeHamiltonian()
HE = calcium_subset.genE1Hamiltonian(1.0) #1 V/cm E field
HM = calcium_subset.genM1Hamiltonian(1.0) #1 Gauss B field

H = H0+HE+HM
eigvals,eigvecs = calcium_subset.diagonalizeHamiltonian(H)

print(eigvals)
print(np.diag(eigvecs)) #should be all 1s
