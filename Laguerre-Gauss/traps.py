import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import numpy.random as rnd
from scipy.stats import chi

width_LG = 40   #µm
k_B = 1.38*10**-23   #µm^2 Kg ms^-2 µK^-1
m_Rb = 85*1.6*10**-27    #kg
h = 6.62*10**-25   # µm^2 kg ms^-1
omega_l = 2*np.pi*(3*10**5)/(1064*10**-9)  #ms^-1
m_e = 9.1*10**-31   #kg
beta = (1/137)* h /(m_e * omega_l**2)   #proportionality factor b/w energy & intensity, µm^2 ms

def LG(x, w, P):      #J ms^-1 µm^-2
    return P*(4*(x**2)/(np.pi*w**4)*np.exp(-2*x**2/w**2))
def E_LG(x, w, P):       #J
    return LG(x, w, P)*beta
def dLG(x, w, P):     #J ms^-1 µm^-3
    return P*(8*x/(np.pi*w**4)*np.exp(-2*x**2/w**2))*(1-2*x**2/w**2)
def dE_LG(x, w, P):      # J µm^-1
    return dLG(x, w, P)*beta

class Atoms_LG(object):

    def __init__(self, n_atoms, temp, w_g, x_0, p_0, freq, A, t):
        """Create an atom"""
        self.n_atoms = n_atoms
        self.temperature = temp
        self.velocities = np.zeros(n_atoms)
        Chi = chi.rvs(df = 2, loc = 0, scale= np.sqrt(k_B*temp/m_Rb), size = n_atoms)    #in µm ms^-1
        for i in range(n_atoms):
            temp = rnd.random()
            if temp<0.5:
                self.velocities[i] = Chi[i]
            else:
                self.velocities[i] = -Chi[i]

        self.x_positions = np.random.normal(x_0, w_g/2, n_atoms)     #in µm
        self.avg_power = p_0
        self.power = p_0

        self.y_positions = E_LG(self.x_positions, width_LG, self.power)/(k_B*10**-6)   #kB in m^2 Kg s^-2 µK^-1
        self.freq = freq
        self.amp = A/100
        self.forces = -dE_LG(self.x_positions, width_LG, self.power)    #J µm^-1
        self.time = t

    def step(self, dt):
        self.x_positions += dt * self.velocities
        self.y_positions = E_LG(self.x_positions, width_LG, self.power)/(k_B*10**-6)
        self.velocities += dt * (10**6)*self.forces/m_Rb     #10**6 to convert J=kg*m^2*s^-2 to kg*µm^2*ms^-2
        self.forces = -dE_LG(self.x_positions, width_LG, self.power)
        self.power += self.amp*self.avg_power*(2*np.pi*self.freq)*np.cos(2*np.pi*self.freq*self.time)*dt
        self.time += dt

def animation_LG(N_at, temperature, width_blue, init_offset, P_0, freq_modulation, amp_modulation):

    X = np.linspace(-80,80,161)
    atoms = Atoms_LG(N_at,temperature, width_blue, init_offset, P_0, freq_modulation, amp_modulation, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], 'ro', ms=5)
    trap, = plt.plot([], [])
    ax.set_xlim(-80, 80)
    ax.set_ylim(0, 1.5*np.max(E_LG(X, width_LG, atoms.power)/(k_B*10**-6)))
    plt.xlabel('Position (µm)')
    plt.ylabel('Energy (µK)')

    def fig_update_LG(i):
        atoms.step(0.001)   #time step, in ms; 1 µs steps
        line.set_data(atoms.x_positions, atoms.y_positions)
        trap.set_data(X, E_LG(X, width_LG, atoms.power)/(k_B*10**-6))
        ax.set_title('t = ' + str(np.round(atoms.time, 2)) + ' ms')
        return line, trap,

    anim = animation.FuncAnimation(fig, fig_update_LG, interval=2, blit=False)

# def trajectory_LG(tot_time):
#
#     steps = int(tot_time/0.0001)
#     times = np.linspace(0, tot_time, steps)
#     positions = np.zeros(steps)
#
#     atoms = Atoms_LG(1,  0, 0.1, 25, 6*10**-3, 0, 1, 0)
#
#     for j in range(int(tot_time/0.0001)):
#         positions[j] = atoms.x_positions[0]
#         atoms.step(0.0001)
#
#     plt.figure()
#     plt.plot(times, positions)
#
#     oscillations = np.fft.rfft(positions)
#     plt.figure()
#     plt.plot(np.log(oscillations[:1000]))




def harm(x, w, P):      #J ms^-1 µm^-2
    return np.heaviside(np.sqrt((25/x)**2)-1, 0.5)*(P)*4*(x**2)/(np.pi*w**4)
def E_harm(x, w, P):       #J
    return LG(x, w, P)*beta
def dharm(x, w, P):     #J ms^-1 µm^-3
    return np.heaviside(np.sqrt((25/x)**2)-1, 0.5)*P*8*x/(np.pi*w**4)
def dE_harm(x, w, P):      # J µm^-1
    return dLG(x, w, P)*beta

class Atoms_harm(object):

    def __init__(self, n_atoms, temp, w_g, x_0, p_0, freq, A, t):
        """Create an atom"""

        self.n_atoms = n_atoms
        self.temperature = temp
        self.velocities = np.zeros(n_atoms)
        Chi = chi.rvs(df = 2, loc = 0, scale= np.sqrt(k_B*temp/m_Rb), size = n_atoms)    #in µm ms^-1
        for i in range(n_atoms):
            temp = rnd.random()
            if temp<0.5:
                self.velocities[i] = Chi[i]
            else:
                self.velocities[i] = -Chi[i]
        self.x_positions = np.random.normal(x_0, w_g/2, n_atoms)     #in µm
        self.avg_power = p_0
        self.power = p_0
        self.y_positions = E_harm(self.x_positions, width_LG, self.power)/(k_B*10**-6)   #kB in m^2 Kg s^-2 µK^-1
        self.freq = freq
        self.amp = A/100
        self.forces = -dE_harm(self.x_positions, width_LG, self.power)    #J µm^-1
        self.time = t


    def step(self, dt):
        self.x_positions += dt * self.velocities
        self.y_positions = E_harm(self.x_positions, width_LG, self.power)/(k_B*10**-6)
        self.velocities += dt * (10**6)*self.forces/m_Rb     #10**6 to convert J=kg*m^2*s^-2 to kg*µm^2*ms^-2
        self.forces = -dE_harm(self.x_positions, width_LG, self.power)
        self.power += self.amp*self.avg_power*(2*np.pi*self.freq)*np.cos(2*np.pi*self.freq*self.time)*dt
        self.time += dt

def animation_harm(N_at, temperature, width_blue, init_offset, P_0, freq_modulation, amp_modulation):

    X = np.linspace(-80,80,161)
    atoms = Atoms_harm(N_at,temperature, width_blue, init_offset, P_0, freq_modulation, amp_modulation, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], 'ro', ms=5)
    trap, = plt.plot([], [])
    ax.set_xlim(-80, 80)
    ax.set_ylim(0, 1.5*np.max(E_harm(X, width_LG, atoms.power)/(k_B*10**-6)))
    plt.xlabel('Position (µm)')
    plt.ylabel('Energy (µK)')

    def fig_update_harm(i):
        atoms.step(0.001)   #time step, in ms; 1 µs steps
        line.set_data(atoms.x_positions, atoms.y_positions)
        trap.set_data(X, E_harm(X, width_LG, atoms.power)/(k_B*10**-6))
        ax.set_title('t = ' + str(np.round(atoms.time, 2)) + ' ms')
        return line, trap,

    anim = animation.FuncAnimation(fig, fig_update_harm, interval=2, blit=False)
