# Atomic Physics - classes and functions
# written by Shira Jackson, 2018
########################################################################

import numpy as np
import copy

from fractions import Fraction

from scipy.constants import pi, hbar, c, e, m_e, epsilon_0, Boltzmann, alpha, physical_constants
kC = 1/(4*pi*epsilon_0)
a0 = physical_constants["atomic unit of length"][0]
Z0 = physical_constants["characteristic impedance of vacuum"][0]
kB = Boltzmann

alpha_au = 4*pi*epsilon_0*a0**3        # 1 a.u. * (1 V/cm)**2 /h in Hz
E_h = alpha**2 * m_e * c**2  # Hartree - unit of energy in au
omega_au = E_h/hbar  # au frequency
mu_B = physical_constants["Bohr magneton"][0]

electric_field_au = E_h/(e*a0)

from sympy.physics.wigner import wigner_3j, wigner_6j
import scipy.linalg as lg

# in au: electric field has units of E_h/(ea_0)

## Atomic State ##

class AtomicState:
    """ class for holding all the quantum numbers and energy value for an atomic level 
    
    Attributes
    ----------
j
    label : str {''}
        Label for the atomic level
    energy : float {0}
        Atomic level energy in units of 2*pi*GHz
    L : float {0}
        Orbital angular momentum quantum number
    S : float {0}
        Spin quantum number
    J : float {0}
        L + S quantum number
    mJ : float {0}
        magnetic sublevel quantum number
    F : float {0}
        L+S+I total angular momentum quantum number
    mF : float {0}
        magnetic sublevel quantum number for hyperfine level
    I : float {0}
        nuclear spin
    parity : int {1}
        +1 for even parity state, -1 for odd parity state
    """
    def __init__(self,label='',energy=0,L=0,S=1.,J=0,mJ=0,F=0,mF=0,I=0,parity=1):
        self.label = label
        self.energy = energy
        self.J = J
        self.mJ = mJ
        self.L = L
        self.S = S
        self.F = F
        self.mF = mF
        self.I = I
        self.P = parity
    
def subLevels(atomicState):
	"""makes copies of the AtomicState with all the possible mJ values, returns them in an array"""
	mLevels = []
	for m in np.arange(-atomicState.J, atomicState.J+1):
		mState = copy.deepcopy(atomicState)
		mState.mJ = m
		mLevels = mLevels + [mState]
	return mLevels	

def hyperfineLevels(atomicState,nuclear_spin):
    """makes copies of the AtomicState with all the possible F values, returns them in an array"""

    fLevels = []
    j_val = atomicState.J

    #for f_val in np.arange(int(abs(j_val-nuclear_spin)),int(j_val+nuclear_spin)+1):
    for f_val in np.arange(abs(j_val-nuclear_spin),j_val+nuclear_spin+1):
        fState = copy.deepcopy(atomicState)
        fState.F = f_val
        #fState.energy = atomicState.energy+hyperfineEnergyShift(atomicState,nuclear_spin)
        fLevels = fLevels + [fState]
    return fLevels	


def subLevelsF(atomicState):
    """makes copies of the AtomicState with all the possible mF values, returns them in an array"""
    mLevels = []
    for m in np.arange(-atomicState.F, atomicState.F+1):
        mState = copy.deepcopy(atomicState)
        mState.mF = m
        mLevels = mLevels + [mState]
    return mLevels	



class AtomicLine:
    """ class for holding information regarding E1 transitions between atomic levels
    
    Attributes
    ----------
    lowerlevel: str {''}
        The initial atomic state label
    upperlevel: str{''}
        The final atomic state label
    fik: float {0}
        the (dimensionless) oscillator strength corresponding to the transition from lowerlevel (i) to upperlevel (k)
    """
    def __init__(self,lowerlevel='',upperlevel='',fik=0):
        self.lowerlevel=lowerlevel
        self.upperlevel=upperlevel
        self.fik=fik


## Conversions & Other useful functions ##


def wavenumberToGHz(wavenum):
    """Convert wavenumber in cm^-1 (ex. from NIST) into frequency in GHz
    """     
    return c*(wavenum*100.)*1e-9      # GHz

def angstromToAU(angstrom):
    """ convert polarizability in angstroms^3 to atomic units """
    return angstrom*1e-30/(a0**3)

def AUtoAngstrom(au):
    """ convert polarizability in angstroms^3 to atomic units """
    return au*1e30*(a0**3)

def SItoAu(si):
    """ convert polarizability in Hz/(V/cm)^2 to atomic units """
    return (2*pi*hbar)*si*1e-4/alpha_au

def DipoletoGf(dipole,wavenum):
    """ convert reduced dipole matrix element in units of e*a0 into weighted oscillator strength gf 
        note: gf = (2J_i + 1)*fik = -(2J_k+1)*fki
        omega in units of 2*pi*GHz
        wavenum in cm^-1"""
    wavenum = wavenum*100.
    #return (2./3.)*(m_e*omeg/hbar)*(dipole*e*a0)**2/(e**2)
    return (2./3.)*(wavenum*a0/alpha)*(dipole**2)

def mFPolarizability(state,I,scalar,tensor):
    """ returns polarizability of mF sublevel from given scalar and tensor polarizabilities """
    F = state.F
    J = state.J
    B_num = F*(2*F-1)*(2*F+1)*(2*J+3)*(J+1)*(2*J+1)
    B_den = (2*F+3)*(F+1)*J*(2*J-1)
    B = B_num/B_den
    A = (3*state.mF**2-F*(F+1))/(F*(2*F-1))
    X = (-1)**(I+J+F) * wigner_6j(F,J,I,J,F,2) * A * np.sqrt(B)    
    
    return scalar + X*tensor


def printLevelData(atomic_level):
    """Display the level information for an atomic state"""
    print("Level: %8s" %(atomic_level.label))
    print("Energy: %.3f GHz,  J = %.1f,  mJ = %.1f,   F = %.1f,   mF = %.1f" %(atomic_level.energy/(2.*pi),atomic_level.J,atomic_level.mJ,atomic_level.F,atomic_level.mF))


def Q(theta):
    """Electric field polarization parameters -1 <= Q <= 1 from polarization angle in degrees
    
    theta = 0 denotes x polarized light
    Q = -1 denotes x polarized light
    
    theta = 90 denotes z polarized light
    Q = +1 denotes z polarized light
    """
    theta_rad = theta*(pi/180.0)
    return 1-2*(np.sin(theta_rad))**2

def Theta(Q):
    """ electric field polarization angle in degrees from Q parameter 
    
    theta = 0 denotes x polarized light
    Q = -1 denotes x polarized light
    
    theta = 90 denotes z polarized light
    Q = +1 denotes z polarized light
    """
    sintheta = np.sqrt((1.0-Q)/2.0)
    return np.arcsin(sintheta)*(180.0/pi)


## Atomic Data ##

# Don't forget: energies are in units of 2*pi*GHz

class AtomicSys:
    def __init__(self,fname_levels,fname_lines,subset=100,hyperfine=False,I=0):
        """Populates array of atomic levels and array of atomic lines 
        
        Parameters
        ------------
        fname_levels: str, filename of txt file containing atomic level data
        fname_lines: str, filename of txt file containing atomic line data
        subset: take a the first subset number of atomic levels in fname_levels
        hyperfine: if True, atomic basis will include hyperfine levels
        I: nuclear spin
        
        
        Fname_levels.txt should be a tab separated list with the following format:
        Configuration	Term		Parity	J	Level(cm-1)
        
        ex.
        4s4p	   3P	-1	0	15157.901
        
        Fname_lines.txt should be a tab separated list with the following format:
        f_ik   	LowerLevel i-Configuration     LowerLevel-Term     LowerLevel-J		UpperLevel k-Confifguration     LowerLevel-Term    LowerLevelJ		
        
        ex.
        1.30E-02    4s2     1S      0       4s8p        1P      1
        """
        self.lines = []
        self.states = []
        self.basis = []
        self.I = I
        self.hyperfine = hyperfine
    
        levels_data = np.genfromtxt(fname_levels,filling_values = 0,invalid_raise = None,dtype=str,skip_header=1,usecols=(0,1,3,4,5))
    
        states = []
    
    
        for s in range(len(levels_data)):
            label = levels_data[s][0]+levels_data[s][1]+str(levels_data[s][3])
            parity = int(levels_data[s][2])
            wavenumber = float(levels_data[s][4])
            J = float(Fraction(levels_data[s][3]))
            S = (int(levels_data[s][1][0])-1)/2
            L_string = levels_data[s][1][1]
            L=10
            if (L_string=='S'): L=0
            elif (L_string=='P'): L=1
            elif(L_string=='D'): L=2
            elif(L_string=='F'): L=3
            elif(L_string=='G'): L=4
            elif(L_string=='H'): L=5
        
            new_state = AtomicState(label=label,energy = 2*pi*wavenumberToGHz(wavenumber),L=L,S=S,J=J,parity=parity)
            states.append(new_state)
    
    # take a subset
        states = states[:subset]
    
    
    
        if(hyperfine):
            f_levels = []
            for s in states: f_levels += hyperfineLevels(s,self.I)
            
            basis = []
            for s in f_levels: basis += subLevelsF(s)
            
        else:
            basis = []
            for s in states: basis += subLevels(s)
    
        lines=[]
        try:
            fik_data = np.genfromtxt(fname_lines, filling_values = 0,invalid_raise = None,dtype=str,skip_header=2)
        
            for i in range(len(fik_data)):
                lower_label = fik_data[i][1]+fik_data[i][2]+(fik_data[i][3])
                upper_label = fik_data[i][4]+fik_data[i][5]+(fik_data[i][6])
                f_value = float(fik_data[i][0])
                new_line = AtomicLine(lower_label,upper_label,f_value)
                lines.append(new_line)
                
        except:
            print("Error with line data")

        self.lines = lines
        self.levels = states
        self.basis = basis


    def state(self,label,m=0):
        try:
            state_list = [x for x in self.basis if (x.label==label)]
            s = [x for x in state_list if (int(x.mJ)==int(m))][0]
            return s
        except:
            print("State not found")
         

    def E1reducedMatrix(self,A,B):
        """E1 reduced matrix element: <A||er||B> in units of e a0
            symmetric oscillator strength gf = (2J_b +1)*f_ba = (2J_a)+1)*f_ab
            
        Parameters
        -----------
        A,B : Instances of AtomicState    
        """	
        N = len(self.basis)
        k = (abs(A.energy-B.energy)*1e9)/c # in m^-1
        gf = 0 
    
    
        for i in range(len(self.lines)):
            if(self.lines[i].lowerlevel == B.label and self.lines[i].upperlevel == A.label):
                gf = (2.*B.J+1)*np.abs(self.lines[i].fik)
                
            if(self.lines[i].lowerlevel == A.label and self.lines[i].upperlevel == B.label):
                gf = (2.*A.J+1)*np.abs(self.lines[i].fik)
        
        ans=0
        if (k != 0): ans = np.sqrt( (3./2.) * (alpha/(k*a0)) * gf ) #some of the higher levels are degenerate to the level of precision reported by NIST
        #if(ans != 0): print ans
        return ans


    def E1Moment(self,A,B,q=0):
        # A, B, are subLevels
        """E1 matrix element: <A|er|B>
        Units are e a0 
        Uses the symmetric Wigner/Racah convention
        
        Parameters
        -----------
        A,B : Instances of AtomicState
        q: spherical tensor component
        """
        if (A.P*B.P == -1):	# Checking to see if the two states have opposite parity 
            return (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * self.E1reducedMatrix(A,B) # Wigner/Racah convention
            #return (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * np.sqrt(2.*B.J+1) * E1reducedMatrix(A,B)  #common non-symmetric convention
            
        else: return 0

    def E1MomentHyperfine(self,A,B,q=0):
        #A, B, are F sublevels
        """E1 matrix element: <A|er|B>
        Units are e a0 
        
        Parameters
        -----------
        A,B : Instances of AtomicState
        q: spherical tensor component"""
        if (A.P*B.P == -1):	# Checking to see if the two states have opposite parity
            return (-1)**(A.F-A.mF) * wigner_3j(A.F,1,B.F,-A.mF,q,B.mF) * (-1)**(B.F+A.J+self.I+1) * np.sqrt((2*A.F+1)*(2*B.F+1)) * wigner_6j(B.J,B.F,self.I,A.F,A.J,1)*self.E1reducedMatrix(A,B)
        
        else: return 0
            
    def M1Moment(self,A,B,q=0):
        """M1 matrix element: <A|mu|B>
            Units are mu_B """
        if (A.P*B.P == +1):	# check to see if the two states have same parity
            return (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * self.M1ReducedMatrix(A,B)
        else: return 0        
        
    def M1ReducedMatrix(self,A,B,gL=-0.9995,gS=-2.0023193):
        """J-reduced matrix element <J|| mu ||J'>"""
        if (A.L==B.L) and (A.S==B.S):
            return np.sqrt((2*A.J+1)*(2*B.J+1)) * ( (-1)**(A.L+A.S+B.J+1) * wigner_6j(A.J,1,B.J,B.L,A.S,A.L) * gL * np.sqrt(A.L*(A.L+1)*(2*A.L+1)) + (-1)**(B.L+B.S+A.J+1) * wigner_6j(A.J,1,B.J,B.S,A.L,A.S) * gS * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) ) 
        else: return 0
        
        
    def M1MomentHyperfine(self,A,B,q=0):
        """M1 matrix element: <A|mu|B>
            Units are mu_B """
        if (A.P*B.P == +1) and (A.L==B.L) and (A.S==B.S):	# Checking to see if the two states have same parity
            return (-1)**(A.F-A.mF) * wigner_3j(A.F,1,B.F,-A.mF,q,B.mF) * self.M1reducedMatrixJ(A,B)
        else: return 0
    
    # I think there is a mistake in this:
    # def M1reducedMatrix(self,A,B):
    #     """Reduced matrix element <J'|| mu ||J> in units of mu_B"""
    #     #if (A.label==B.label): this gives zero for triplets which isn't right
    #     return (-1)**(A.L+A.S+B.J+1) * np.sqrt( (2*A.J+1)*(2*B.J+1) ) * ( 0.9995 * wigner_6j(A.J,1.,B.J,B.L,A.S,A.L) * np.sqrt(A.L*(A.L+1)*(2*A.L+1)) + 2.003 * wigner_6j(A.J,1.,B.J,B.S,A.L,A.S) * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) ) 
    #     #else: return 0
    #     # A.S == B.S, since dipole operator doesn't touch spin
        
    def M1reducedMatrixHyperfine(self,A,B):
        """Reduced matrix element <J'|| mu ||J>"""
        # # # Fill this in and test it # # #
        if (A.label==B.label): 
            return 0
        else: return 0

    def genFreeHamiltonian(self):
        """ Generates the free Hamiltonian matrix in units of 2*pi*GHz """
        N = len(self.basis)	
        H0 = np.matrix(np.diag( [self.basis[i].energy for i in range(N)] ))
        return H0
    
    def genE1Operators(self,hyperfine=False):
        """ Generates the E1 dipole moment operator matrices in units of e*a0."""
        N = len(self.basis)
        
        # Dipole moment operator matrices
        # 	only the dipole lowering part of the operator is defined here
        Dz = np.matrix(np.zeros( (N,N) ))
        Dplus = np.matrix(np.zeros( (N,N) ))
        Dminus = np.matrix(np.zeros( (N,N) ))

        for i in range(N):
            for j in range(i+1,N):		
                # Dipole moment operators
                if hyperfine:
                    dz = self.E1MomentHyperfine(self.basis[i],self.basis[j],q=0)
                    dplus = self.E1MomentHyperfine(self.basis[i],self.basis[j],q=+1)
                    dminus = self.E1MomentHyperfine(self.basis[i],self.basis[j],q=-1)	
                
                else:
                    dz = self.E1Moment(self.basis[i],self.basis[j],q=0)
                    dplus = self.E1Moment(self.basis[i],self.basis[j],q=+1)
                    dminus = self.E1Moment(self.basis[i],self.basis[j],q=-1)	

                Dz[i,j],Dplus[i,j],Dminus[i,j] = dz,dplus,dminus

        Dx = (Dminus - Dplus)/np.sqrt(2)
        Dy = (Dminus + Dplus)/(1j*np.sqrt(2))
    
        return Dz,Dplus,Dminus
    
    def genDiagM1(self):
        """Generates the diagonal magnetic moment matrix in units of u_B. """
        N = len(self.basis)
        
        Mz_list = [self.M1Moment(self.basis[i],self.basis[i],q=0) for i in range(N)]
        Mz_list_float = [float(i) for i in Mz_list] # evaluates all the sqrts from the wigner symbols
        diagMz = np.diag(Mz_list_float)	# diagonal magnetic moments
        
        return diagMz
    
    def genM1Operators(self,hyperfine=False):
        """ Generates the M1 dipole moment operator matrices in units of mu_B."""
        N = len(self.basis)
        
        # Dipole moment operator matrices
        # 	only the dipole lowering part of the operator is defined here
        
        Mz = np.matrix(np.zeros( (N,N) ))
        Mplus = np.matrix(np.zeros( (N,N) ))
        Mminus = np.matrix(np.zeros( (N,N) ))

        
        for i in range(N):
            for j in range(i+1,N):		
                # Dipole moment operators
                if hyperfine:
                    mz = self.M1MomentHyperfine(self.basis[i],self.basis[j],q=0)
                    mplus = self.M1MomentHyperfine(self.basis[i],self.basis[j],q=+1)
                    mminus = self.M1MomentHyperfine(self.basis[i],self.basis[j],q=-1)
                else:
                    mz = self.M1Moment(self.basis[i],self.basis[j],q=0)
                    mplus = self.M1Moment(self.basis[i],self.basis[j],q=+1)
                    mminus = self.M1Moment(self.basis[i],self.basis[j],q=-1)

                Mz[i,j],Mplus[i,j],Mminus[i,j] = mz,mplus,mminus

        Mx = (Mminus - Mplus)/np.sqrt(2)
        My = (Mminus + Mplus)/(1j*np.sqrt(2))
    
        return Mz,Mplus,Mminus

    def genM1Hamiltonian(self,Bz):
        """ returns the M1 interaction Hamiltonian matrix in units of 2*pi*GHz 
            Bz in units of Gauss """
        mz,mplus,mminus = self.genM1Operators()
        diagmz = self.genDiagM1()
        H_int_M1 = -(mu_B/hbar) * (mz + mz.H + diagmz) *Bz* 1e-4 *1e-9
        return H_int_M1
    
    def genE1Hamiltonian(self,Ez):
        """ returns the E1 interaction Hamiltonian matrix in units of 2*pi*GHz 
            E_z in units of V/cm """
        dz,dplus,dminus = self.genE1Operators()
        H_int_E1 = -(e*a0/hbar) * (dz + dz.H) * ((Ez*100)/2.) *1e-9	#2*pi*GHz    #E/2 because E is divided between co-rotating and counter-rotating excitations 
        return H_int_E1
        
    def diagonalizeHamiltonian(self,H):
        """ returns eigenvalues and eigenvectors of Hamiltonian H """
        eig_vals,eig_vecs = (lg.eigh(H)) # new eigenvalues

        newbasisorder=[]
        
        for i in range(len(eig_vecs)):
            basisIndex = np.abs(eig_vecs[:,i]).argmax() # IMPORTANT: note how the eigenvector is picked out - eigenvectors are zeros with a single one at some index
            newbasisorder.append(basisIndex)
        
        eig_vecs = [x for _,x in sorted(zip(newbasisorder,eig_vecs))]
        
        return eig_vals,eig_vecs

    def calculateAlpha(self,atomic_state,omega=0.0001,Q=1.0,phi=0,prnt=False,hyperfine=False):
        """returns polarizability of atomic_state in atomic units 
        Parameters
        -----------
        atomic_state: Instance of AtomicState
        omega: Electric field frequency in units of 2*pi*GHz
        Q: Electric field polarization parameters -1 <= Q <= 1
        phi: Phase difference between orthogonal electric field components
        prnt: If True will print the atomic level and calculated polarizability   
        """
        sum1 = 0
        sum2 = 0
        
        phase_x = np.exp(1j*phi)
        
        E_x = np.sqrt((1-Q)/2)
        E_z = np.sqrt((1+Q)/2)
    
        for j in range(len(self.basis)):
            
            energy_denominator =  self.basis[j].energy - atomic_state.energy
            
            if energy_denominator != 0:
                
                if hyperfine:
                    dz = self.E1MomentHyperfine(atomic_state,self.basis[j],q=0)
                    dplus = self.E1MomentHyperfine(atomic_state,self.basis[j],q=+1)
                    dminus = self.E1MomentHyperfine(atomic_state,self.basis[j],q=-1)	
                
                else:
                    dz = self.E1Moment(atomic_state,self.basis[j],q=0)
                    dplus = self.E1Moment(atomic_state,self.basis[j],q=+1)
                    dminus = self.E1Moment(atomic_state,self.basis[j],q=-1)	
        
                if energy_denominator >=0:
                    dz,dplus,dminus = -dz,-dplus,-dminus
                
                dx = (dminus - dplus)/np.sqrt(2)
                matrix_element = (E_z*dz+E_x*phase_x*dx)*e*a0 
                
                sum1 += np.abs(matrix_element)**2/(hbar*1e9*(energy_denominator-omega))
                sum2 += np.abs(matrix_element)**2/(hbar*1e9*(energy_denominator+omega))
            
                
        pol = (sum1+sum2)/alpha_au 
                
    
        if prnt: 
        
            if hyperfine:
                print ("%s F = %i mF = %i  Polarizability: %.3f" %(atomic_state.label,atomic_state.F,atomic_state.mF,pol))
            else:
                print ("%s J = %i mJ = %i  Polarizability: %.3f" %(atomic_state.label,atomic_state.J,atomic_state.mJ,pol))
                
        return pol


    def transitionProbability(self,ground_state,excited_state):
        """
        Calculate the radiative lifetime A_ki in s^-1 for E1 (electric dipole) transitions
        i = lower state, k = upper state
        
        for single states, not multiplets
        
        """
        
        A = ground_state
        B = excited_state
                    
        
        k = (abs(A.energy-B.energy)*1e9)/c # in m^-1
        lam = 2*pi/k #in m
        ggf = 0
    
        for i in range(len(self.lines)):
            if(self.lines[i].lowerlevel == B.label and self.lines[i].upperlevel == A.label):
                ggf = (2.*B.J+1)*np.abs(self.lines[i].fik)/(2.*A.J+1)
                
            if(self.lines[i].lowerlevel == A.label and self.lines[i].upperlevel == B.label):
                ggf = (2.*A.J+1)*np.abs(self.lines[i].fik)/(2.*B.J+1)
        
        fact = 2*pi*e**2/(m_e*c*epsilon_0*lam**2)
        
        Aki = fact*ggf
        if A.energy-B.energy>0:
            Aki = -Aki
        
        return Aki
        
    def radiativeLifetime(self,state_k,prnt=False):
        """
        Calculate the lifetime of a state  - note: this only uses E1 transitions! If a forbidden transition is an important decay path for this state, this calculation will not be accurate. 
        """
        N = len(self.basis)
        
        total_probability = 0
        
        SLJ_done = []
        
        for i in range(N):
            # we don't want to sum over ALL states SLJm, only over the states SLJ?
            state_i = self.basis[i]
            if (state_i.energy<state_k.energy) and (state_i.label not in SLJ_done):
                Aki = self.transitionProbability(state_i,state_k)
                total_probability += Aki
                SLJ_done.append(state_i.label)
                
                if(prnt and Aki!=0):
                    print("Decay to state %s: %.2f per second"%(state_i.label,Aki))
                
        lifetime = 1/total_probability
        return lifetime
            
        
    def branchingRatio(self,excited_state,ground_state):
        """Returns the branching ratio of decay from the excited state to the ground state"""
        lifetime = self.radiativeLifetime(excited_state)
        
        Aki = self.transitionProbability(ground_state,excited_state)
        
        return Aki*lifetime
        

    def twoPhotonE1Moment(self,ground_state,excited_state,Q=1.0,phi=0,prnt=False,):
        """
        Q: Electric field polarization parameters -1 <= Q <= 1
        phi: Phase difference between orthogonal electric field components
        
        returns two-photon E1 moment in units of (e*a0)^2/E_h 
        
        two photon E1 moment is calculated as the sum of terms i:
        
        <e|D|i><i|D|g>
        ______________
        E(laser) - E(i-g)
        
        to get the Rabi frequency in s^-1: multiply this by electric_field^2/hbar
        
        """
        sum = 0
        
        phase_x = np.exp(1j*phi)
        
        E_x = np.sqrt((1-Q)/2)
        E_z = np.sqrt((1+Q)/2)
    
        omega_0 = (excited_state.energy - ground_state.energy)/2.0
        
        for j in range(len(self.basis)):
            
            energy_denominator =  hbar*1e9*(omega_0 - self.basis[j].energy)/E_h #units of E_h
            
            if energy_denominator != 0:
                            
                dz1 = self.E1Moment(excited_state,self.basis[j],q=0)
                dplus1 = self.E1Moment(excited_state,self.basis[j],q=+1)
                dminus1 = self.E1Moment(excited_state,self.basis[j],q=-1)	
        
                dz2 = self.E1Moment(self.basis[j],ground_state,q=0)
                dplus2 = self.E1Moment(self.basis[j],ground_state,q=+1)
                dminus2 = self.E1Moment(self.basis[j],ground_state,q=-1)	
                
                if energy_denominator >=0:
                    dz1,dplus1,dminus1 = -dz1,-dplus1,-dminus1
                    dz2,dplus2,dminus2 = -dz2,-dplus2,-dminus2
                
                dx1 = (dminus1 - dplus1)/np.sqrt(2)
                dx2 = (dminus2 - dplus2)/np.sqrt(2)
    
                matrix_element1 = (E_z*dz1+E_x*phase_x*dx1)
                matrix_element2 = (E_z*dz2+E_x*phase_x*dx2)
                

                sum += (matrix_element1*matrix_element2)/(energy_denominator)
            
        return sum

