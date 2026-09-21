import numpy as np
from model_potential import CO_morse

class MD:
    """
    Molecular Dynamics class for CO molecule trajectory propagation.
    Uses velocity Verlet algorithm for integration.
    """
    
    # Conversion factors
    FORCE_CONVERSION = 0.009648      # eV/Å·amu to Å/fs²
    ENERGY_CONVERSION = 103.650907   # amu·Å²/fs² to eV
    
    def __init__(self, x, v, m, dt, calculator):
        """
        Initialize MD simulation.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Initial positions [xC, xO] in Angstroms
        v : array-like, shape (2,)
            Initial velocities [vC, vO] in Angstrom/fs
        m : array-like, shape (2,)
            Masses [mC, mO] in amu
        dt : float
            Time step in fs
        calculator : class
            Calculator class (CO_morse) for energy and force calculations
        """
        self.x = np.array(x, dtype=float)
        self.v = np.array(v, dtype=float)
        self.m = np.array(m, dtype=float)
        self.dt = dt
        self.calculator_class = calculator
        
        # Create calculator instance
        self.calc = self.calculator_class(self.x, self.v, self.m)
    
    def velocity_verlet(self):
        """
        Perform one velocity Verlet integration step.
        
        The velocity Verlet algorithm:
        1. x(t+h) = x(t) + v(t)*h + 0.5*a(t)*h²
        2. v(t+h) = v(t) + 0.5*[a(t+h) + a(t)]*h
        
        where a = F/m with proper unit conversion
        """
        h = self.dt
        
        # Update calculator with current positions and velocities
        self.calc.x = self.x.copy()
        self.calc.v = self.v.copy()
        
        # Calculate forces at current positions
        self.calc.calculate_energy_and_forces()
        F_old = self.calc.F.copy()
        
        # Convert forces to accelerations (with unit conversion)
        a_old = (F_old / self.m) * self.FORCE_CONVERSION
        
        # Update positions
        self.x = self.x + self.v * h + 0.5 * a_old * h**2
        
        # Calculate forces at new positions
        self.calc.x = self.x.copy()
        self.calc.calculate_energy_and_forces()
        F_new = self.calc.F.copy()
        
        # Convert new forces to accelerations
        a_new = (F_new / self.m) * self.FORCE_CONVERSION
        
        # Update velocities
        self.v = self.v + 0.5 * (a_old + a_new) * h
    
    def run(self, nsteps):
        """
        Run MD simulation for nsteps.
        
        Parameters:
        -----------
        nsteps : int
            Number of integration steps
        
        Returns:
        --------
        tuple : (xC_array, xO_array, Ekin_array, Epot_array)
            Arrays containing trajectory data
        """
        # Initialize arrays to store trajectory
        xC_traj = np.zeros(nsteps + 1)
        xO_traj = np.zeros(nsteps + 1)
        Ekin_traj = np.zeros(nsteps + 1)
        Epot_traj = np.zeros(nsteps + 1)
        
        # Store initial values
        self.calc.x = self.x.copy()
        self.calc.v = self.v.copy()
        self.calc.calculate_energy_and_forces()
        K_initial = self.calc.calculate_kinetic_energy() * self.ENERGY_CONVERSION
        
        xC_traj[0] = self.x[0]
        xO_traj[0] = self.x[1]
        Ekin_traj[0] = K_initial
        Epot_traj[0] = self.calc.V
        
        # Run simulation
        for step in range(nsteps):
            self.velocity_verlet()
            
            # Update calculator and compute energies
            self.calc.x = self.x.copy()
            self.calc.v = self.v.copy()
            self.calc.calculate_energy_and_forces()
            K = self.calc.calculate_kinetic_energy() * self.ENERGY_CONVERSION
            
            # Store trajectory data
            xC_traj[step + 1] = self.x[0]
            xO_traj[step + 1] = self.x[1]
            Ekin_traj[step + 1] = K
            Epot_traj[step + 1] = self.calc.V
        
        return xC_traj, xO_traj, Ekin_traj, Epot_traj


# Test block
if __name__ == "__main__":
    print("Testing MD module...\n")
    
    # Simple test: molecule at equilibrium with zero velocity
    x0 = np.array([0.0, 1.128])
    v0 = np.array([0.0, 0.0])
    m = np.array([12.01, 16.00])
    dt = 0.1
    
    md = MD(x0, v0, m, dt, CO_morse)
    xC, xO, Ekin, Epot = md.run(100)
    
    print(f"Initial bond length: {x0[1] - x0[0]:.6f} Å")
    print(f"Final bond length: {xO[-1] - xC[-1]:.6f} Å")
    print(f"Initial total energy: {Ekin[0] + Epot[0]:.6f} eV")
    print(f"Final total energy: {Ekin[-1] + Epot[-1]:.6f} eV")
    print(f"Energy drift: {abs((Ekin[-1] + Epot[-1]) - (Ekin[0] + Epot[0])):.6e} eV")
    print("\nMD module test complete!")
