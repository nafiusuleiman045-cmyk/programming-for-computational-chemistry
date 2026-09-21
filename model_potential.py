import numpy as np

def morse_potential(r, De=11.22, alpha=2.594, re=1.128):
    """
    Calculate the Morse potential energy and force.
    
    Parameters:
    -----------
    r : float
        Bond length in Angstroms
    De : float
        Well depth in eV (default: 11.22)
    alpha : float
        Width parameter in Angstrom^-1 (default: 2.594)
    re : float
        Equilibrium bond length in Angstroms (default: 1.128)
    
    Returns:
    --------
    tuple : (V, F)
        V: potential energy in eV
        F: force in eV/Angstrom (positive means repulsive, pushing atoms apart)
    """
    exp_term = np.exp(-alpha * (r - re))
    V = De * (1 - exp_term)**2
    
    # Force: F = -dV/dr, but we return the derivative for use in equations
    # The force on the right atom (oxygen) is positive when repulsive
    F = 2 * alpha * De * (1 - exp_term) * exp_term
    
    return V, F


class CO_morse:
    """
    Class to model a CO molecule with Morse potential.
    Molecule constrained to move along x-axis with xO > xC.
    """
    
    def __init__(self, x, v, m=None):
        """
        Initialize CO molecule.
        
        Parameters:
        -----------
        x : array-like, shape (2,)
            Positions [xC, xO] in Angstroms
        v : array-like, shape (2,)
            Velocities [vC, vO] in Angstrom/fs
        m : array-like, shape (2,), optional
            Masses [mC, mO] in amu (default: [12.01, 16.00])
        """
        self.x = np.array(x, dtype=float)
        self.v = np.array(v, dtype=float)
        self.m = np.array(m if m is not None else [12.01, 16.00], dtype=float)
        
        # Initialize energy variables
        self.V = 0.0
        self.K = 0.0
        self.F = np.zeros(2)
    
    def calculate_energy_and_forces(self):
        """
        Calculate bond length, potential energy, and forces on atoms.
        
        Returns:
        --------
        None (updates self.V and self.F)
        """
        # Calculate bond length (use absolute value to handle any ordering)
        r = abs(self.x[1] - self.x[0])
        
        # Determine direction: if xO > xC, sign is positive; otherwise negative
        sign = 1.0 if self.x[1] > self.x[0] else -1.0
        
        # Get potential energy and force from Morse potential
        self.V, F_magnitude = morse_potential(r)
        
        # Forces on atoms (Newton's third law: equal and opposite)
        # The sign ensures forces point in the correct direction
        self.F[0] = -F_magnitude * sign  # Force on carbon
        self.F[1] = F_magnitude * sign   # Force on oxygen
    
    def calculate_kinetic_energy(self):
        """
        Calculate total kinetic energy of the molecule.
        
        Returns:
        --------
        float : kinetic energy in amu*Angstrom^2/fs^2
        """
        self.K = 0.5 * np.sum(self.m * self.v**2)
        return self.K
    
    def get_total_energy(self):
        """
        Get total energy (potential + kinetic).
        
        Returns:
        --------
        float : total energy
        """
        return self.V + self.K


# Test block
if __name__ == "__main__":
    print("Running sanity checks...\n")
    
    # Test 1: Force at equilibrium should be zero
    print("Test 1: Force at equilibrium (r = re = 1.128 Å)")
    x_eq = np.array([0.0, 1.128])
    v_eq = np.array([0.0, 0.0])
    co_eq = CO_morse(x_eq, v_eq)
    co_eq.calculate_energy_and_forces()
    
    print(f"  Force on C: {co_eq.F[0]:.10f} eV/Å")
    print(f"  Force on O: {co_eq.F[1]:.10f} eV/Å")
    
    if np.abs(co_eq.F[0]) < 1e-10 and np.abs(co_eq.F[1]) < 1e-10:
        print("  ✓ Test 1 PASSED: Forces are zero at equilibrium\n")
    else:
        print("  ✗ Test 1 FAILED: Forces should be zero at equilibrium\n")
    
    # Test 2: FC + FO = 0 for random positions
    print("Test 2: Newton's third law (FC + FO = 0) for random positions")
    np.random.seed(42)
    for i in range(5):
        xC_rand = np.random.uniform(-2, 0)
        xO_rand = np.random.uniform(xC_rand + 0.5, xC_rand + 3.0)
        x_rand = np.array([xC_rand, xO_rand])
        v_rand = np.random.uniform(-1, 1, 2)
        
        co_rand = CO_morse(x_rand, v_rand)
        co_rand.calculate_energy_and_forces()
        
        force_sum = co_rand.F[0] + co_rand.F[1]
        r_test = x_rand[1] - x_rand[0]
        
        print(f"  Trial {i+1}: r = {r_test:.3f} Å, FC + FO = {force_sum:.10e} eV/Å")
        
        if np.abs(force_sum) < 1e-10:
            print(f"    ✓ PASSED")
        else:
            print(f"    ✗ FAILED")
    
    print("\nAll sanity checks complete!")
