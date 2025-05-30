import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

class SafeSlope:
    """
    SafeSlope algorithm implementation based on the paper:
    "A Multi-Fidelity Bayesian Approach to Safe Controller Design"
    by Ethan Lau, Vaibhav Srivastava, and Shaunak D. Bopardikar
    """
    def __init__(self, X, safety_threshold, initial_safe_set, 
                 delta_f=0.1, delta_m=0.1, beta_scaling=None,
                 kernel=None, kernel_slope=None, noise=1e-4):
        """
        Initialize the SafeSlope algorithm.
        
        Args:
            X: Domain of the function to optimize (grid)
            safety_threshold: Safety threshold h
            initial_safe_set: Initial set of safe points S_0
            delta_f: Confidence parameter for function values
            delta_m: Confidence parameter for slopes
            beta_scaling: Function for scaling beta over time (default: t²π²/6)
            kernel: GP kernel for function values
            kernel_slope: GP kernel for slopes
            noise: Measurement noise variance
        """
        self.X = X
        self.h = safety_threshold
        self.S0 = initial_safe_set.copy()
        self.delta_f = delta_f
        self.delta_m = delta_m
        self.noise = noise
        
        # Set default beta scaling if not provided
        if beta_scaling is None:
            self.beta_scaling = lambda t: (t**2) * (np.pi**2) / 6
        else:
            self.beta_scaling = beta_scaling
        
        # Default kernels if not provided
        if kernel is None:
            self.kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=noise)
        else:
            self.kernel = kernel
        
        if kernel_slope is None:
            self.kernel_slope = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=noise)
        else:
            self.kernel_slope = kernel_slope
            
        # Initialize GP models
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0, n_restarts_optimizer=10)
        
        # For slopes, we need a separate GP for each axis
        self.n_dims = X.shape[1]
        self.gp_slopes = [GaussianProcessRegressor(kernel=self.kernel_slope, alpha=0, 
                                                 n_restarts_optimizer=10) for _ in range(self.n_dims)]
        
        # Initialize safe set and data collections
        self.safe_set = self.S0.copy()
        self.X_data = []
        self.y_data = []
        
        # Create incidence matrices for slopes along each axis
        self.create_slope_incidence_matrices()
        
        # Storage for slope bounds
        self.slope_upper_bounds = {}  # Will store (x, x') -> bound
        
    def create_slope_incidence_matrices(self):
        """Create incidence matrices for calculating slopes along each axis."""
        # This is a simplification for grid domains
        # In practice, we would create proper incidence matrices based on the domain structure
        pass
    
    def calculate_beta(self, t):
        """Calculate the exploration parameter beta at time t."""
        num_points = len(self.X)
        beta_f = 2 * np.log(num_points * self.beta_scaling(t) / self.delta_f)
        beta_m = 2 * np.log(num_points * self.n_dims * self.beta_scaling(t) / self.delta_m)
        return beta_f, beta_m
    
    def update_models(self, x, y):
        """Update GP models with a new observation."""
        # Add new data point
        self.X_data.append(x)
        self.y_data.append(y)
        
        # Update function GP
        X_array = np.array(self.X_data)
        y_array = np.array(self.y_data)
        self.gp.fit(X_array, y_array)
        
        # Update slope GPs (simplified for clarity)
        # In practice, we would use the derivative GPs or finite differences
        # This part would need proper implementation based on how slopes are modeled
        
    def update_slope_bounds(self, t):
        """Update the slope bounds at time t."""
        # For each pair of adjacent points, calculate the slope bound
        # This is a simplification - a full implementation would handle
        # the slope calculations and bound updates more rigorously
        
        # Example implementation for grid domains
        beta_f, beta_m = self.calculate_beta(t)
        
        # Loop through all pairs of adjacent points in the safe set
        for x in self.safe_set:
            for dim in range(self.n_dims):
                # Find adjacent point along dimension dim
                x_adj = x.copy()
                x_adj[dim] += self.grid_step  # Assumes uniform grid step
                
                if tuple(x_adj) in self.safe_set:
                    # Calculate slope bound between x and x_adj
                    key = (tuple(x), tuple(x_adj))
                    
                    # Simplified version - would use proper GP slope predictions
                    if key not in self.slope_upper_bounds:
                        self.slope_upper_bounds[key] = float('inf')
                    else:
                        # Update bound based on GP slope prediction
                        # This is where the paper's equation (10) would be implemented
                        pass
                    
    def get_confidence_bounds(self, x, t):
        """Get lower and upper confidence bounds for point x at time t."""
        beta_f, _ = self.calculate_beta(t)
        
        # Get mean and std from GP
        x_reshaped = np.array([x])
        mean, std = self.gp.predict(x_reshaped, return_std=True)
        
        # Calculate bounds
        lcb = mean[0] - np.sqrt(beta_f) * std[0]
        ucb = mean[0] + np.sqrt(beta_f) * std[0]
        
        return lcb, ucb
    
    def is_safe(self, x, x_parent, t):
        """Check if point x is safe given parent point x_parent at time t."""
        # Get upper bound for parent point
        _, parent_ucb = self.get_confidence_bounds(x_parent, t)
        
        # Get slope bound between parent and x
        key = (tuple(x_parent), tuple(x))
        slope_bound = self.slope_upper_bounds.get(key, float('inf'))
        
        # Calculate distance
        dist = np.linalg.norm(np.array(x) - np.array(x_parent))
        
        # Safety condition (equation 11 in the paper)
        return parent_ucb + slope_bound * dist <= self.h
        
    def get_safe_set(self, t):
        """Update the safe set at time t based on current GP models."""
        new_safe_set = self.safe_set.copy()
        
        # For each point in current safe set, check adjacent points
        for x in self.safe_set:
            for dim in range(self.n_dims):
                for step in [-self.grid_step, self.grid_step]:
                    x_adj = x.copy()
                    x_adj[dim] += step
                    
                    # Check if x_adj is in domain and not already in safe set
                    if tuple(x_adj) not in new_safe_set and self.is_in_domain(x_adj):
                        # Check if x_adj is safe
                        if self.is_safe(x_adj, x, t):
                            new_safe_set.add(tuple(x_adj))
        
        self.safe_set = new_safe_set
        return self.safe_set
    
    def is_in_domain(self, x):
        """Check if point x is in the domain."""
        # Implement based on how domain is represented
        # For grid domains, this would check if x is within the grid bounds
        return True  # Placeholder
    
    def get_minimizer_set(self, t):
        """Get the set of points that potentially minimize f."""
        M_t = set()
        
        min_ucb = float('inf')
        for x in self.safe_set:
            _, ucb = self.get_confidence_bounds(x, t)
            min_ucb = min(min_ucb, ucb)
        
        for x in self.safe_set:
            lcb, _ = self.get_confidence_bounds(x, t)
            if lcb <= min_ucb:
                M_t.add(tuple(x))
                
        return M_t
    
    def get_expander_set(self, t):
        """Get the set of points that could expand the safe set."""
        G_t = set()
        
        for x in self.safe_set:
            growth = 0
            
            # Check if sampling x could add points to the safe set
            for dim in range(self.n_dims):
                for step in [-self.grid_step, self.grid_step]:
                    x_adj = x.copy()
                    x_adj[dim] += step
                    
                    if tuple(x_adj) not in self.safe_set and self.is_in_domain(x_adj):
                        lcb, _ = self.get_confidence_bounds(x, t)
                        
                        # Check if lower bound + slope_bound * distance <= h
                        key = (tuple(x), tuple(x_adj))
                        slope_bound = self.slope_upper_bounds.get(key, float('inf'))
                        dist = np.linalg.norm(np.array(x_adj) - np.array(x))
                        
                        if lcb + slope_bound * dist <= self.h:
                            growth += 1
            
            if growth > 0:
                G_t.add(tuple(x))
                
        return G_t
    
    def select_next_point(self, t):
        """Select next point to evaluate based on SafeSlope algorithm."""
        # Get minimizer and expander sets
        M_t = self.get_minimizer_set(t)
        G_t = self.get_expander_set(t)
        
        union_set = M_t.union(G_t)
        if not union_set:
            # If no points in union, select point from safe set with maximum width
            max_width = -float('inf')
            best_point = None
            
            for x in self.safe_set:
                lcb, ucb = self.get_confidence_bounds(x, t)
                width = ucb - lcb
                
                if width > max_width:
                    max_width = width
                    best_point = x
                    
            return best_point
        
        # Select point with maximum width (equation 8 in the paper)
        max_width = -float('inf')
        best_point = None
        
        for x in union_set:
            lcb, ucb = self.get_confidence_bounds(x, t)
            width = ucb - lcb
            
            if width > max_width:
                max_width = width
                best_point = x
                
        return best_point
    
    def optimize(self, f, f_low=None, num_iterations=50):
        """
        Run the SafeSlope optimization algorithm.
        
        Args:
            f: Function to optimize
            f_low: Low-fidelity function (for multi-fidelity mode)
            num_iterations: Number of iterations to run
            
        Returns:
            best_x: The point with minimum function value found
            best_y: Minimum function value found
            X_data: List of points evaluated
            y_data: Function values at evaluated points
        """
        # Initialize with points from S0
        for x in self.S0:
            y = f(x)
            self.update_models(x, y)
        
        # Main optimization loop
        for t in range(1, num_iterations+1):
            # Update safe set
            self.update_slope_bounds(t)
            self.get_safe_set(t)
            
            # Select next point to evaluate
            x_next = self.select_next_point(t)
            
            # Evaluate function at selected point
            y_next = f(x_next)
            
            # Update models
            self.update_models(x_next, y_next)
            
            # Print progress
            if t % 10 == 0:
                print(f"Iteration {t}: f({x_next}) = {y_next}")
        
        # Return best point found
        best_idx = np.argmin(self.y_data)
        best_x = self.X_data[best_idx]
        best_y = self.y_data[best_idx]
        
        return best_x, best_y, self.X_data, self.y_data


class MultiFidelitySafeSlope(SafeSlope):
    """Extension of SafeSlope for multi-fidelity models using the AR-1 model."""
    
    def __init__(self, X, safety_threshold, initial_safe_set, rho=0.5,
                 delta_f=0.1, delta_m=0.1, beta_scaling=None,
                 kernel_high=None, kernel_low=None, kernel_slope=None, 
                 noise_high=1e-4, noise_low=1e-8):
        """
        Initialize the Multi-Fidelity SafeSlope algorithm.
        
        Args:
            X: Domain of the function to optimize (grid)
            safety_threshold: Safety threshold h
            initial_safe_set: Initial set of safe points S_0
            rho: Scaling parameter for AR-1 model
            delta_f, delta_m: Confidence parameters
            beta_scaling: Function for scaling beta over time
            kernel_high, kernel_low, kernel_slope: GP kernels for high/low fidelity functions and slopes
            noise_high, noise_low: Measurement noise for high/low fidelity
        """
        super().__init__(X, safety_threshold, initial_safe_set, 
                         delta_f, delta_m, beta_scaling,
                         kernel_high, kernel_slope, noise_high)
        
        self.rho = rho
        self.noise_low = noise_low
        
        # Add low-fidelity GP
        if kernel_low is None:
            self.kernel_low = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=noise_low)
        else:
            self.kernel_low = kernel_low
            
        self.gp_low = GaussianProcessRegressor(kernel=self.kernel_low, alpha=0, n_restarts_optimizer=10)
        
        # Data storage for low-fidelity samples
        self.X_low_data = []
        self.y_low_data = []
        
    def evaluate_low_fidelity(self, f_low):
        """Evaluate low-fidelity function at all points in the domain."""
        for x in self.X:
            y_low = f_low(x)
            self.X_low_data.append(x)
            self.y_low_data.append(y_low)
            
        # Update low-fidelity GP
        X_array = np.array(self.X_low_data)
        y_array = np.array(self.y_low_data)
        self.gp_low.fit(X_array, y_array)
        
    def predict_ar1(self, x):
        """
        Make a prediction using the AR-1 model:
        f(x) = ρ * f_low(x) + δ(x)
        """
        # Get low-fidelity prediction
        x_reshaped = np.array([x])
        mean_low, _ = self.gp_low.predict(x_reshaped, return_std=True)
        
        # Get high-fidelity correction
        if len(self.X_data) > 0:
            mean_high, std_high = self.gp.predict(x_reshaped, return_std=True)
            
            # AR-1 model prediction
            mean = self.rho * mean_low[0] + mean_high[0]
            std = std_high[0]  # Simplified - in practice we'd calculate this differently
        else:
            # If no high-fidelity data yet, use scaled low-fidelity
            mean = self.rho * mean_low[0]
            std = np.sqrt(self.noise_high)  # Default uncertainty
            
        return mean, std
    
    def get_confidence_bounds(self, x, t):
        """Get lower and upper confidence bounds using the AR-1 model."""
        beta_f, _ = self.calculate_beta(t)
        
        # Get AR-1 prediction
        mean, std = self.predict_ar1(x)
        
        # Calculate bounds
        lcb = mean - np.sqrt(beta_f) * std
        ucb = mean + np.sqrt(beta_f) * std
        
        return lcb, ucb
    
    def optimize(self, f, f_low, num_iterations=50):
        """
        Run the Multi-Fidelity SafeSlope optimization algorithm.
        
        Args:
            f: High-fidelity function to optimize
            f_low: Low-fidelity function
            num_iterations: Number of iterations to run
            
        Returns:
            best_x: The point with minimum function value found
            best_y: Minimum function value found
            X_data: List of points evaluated
            y_data: Function values at evaluated points
        """
        # First, evaluate low-fidelity function at all points
        self.evaluate_low_fidelity(f_low)
        
        # Initialize with points from S0 using high-fidelity function
        for x in self.S0:
            y = f(x)
            self.update_models(x, y)
        
        # Continue with SafeSlope optimization loop
        return super().optimize(f, None, num_iterations)


# Example usage function
def run_example():
    """Reproduce the example from the paper."""
    # Define the true system matrices (from equation 15)
    A_true = np.array([
        [0.785, -0.260],
        [-0.260, 0.315]
    ])
    B_true = np.array([[1.475], [0.607]])
    
    # Define the approximate model matrices (from equation 16)
    A_approx = np.array([
        [0.700, -0.306],
        [-0.306, 0.342]
    ])
    B_approx = np.array([[1.543], [0.524]])
    
    # Define cost function parameters
    Q = np.eye(2)  # Identity matrix for Q
    R = 1.0        # Scalar R
    
    # Initial state
    z0 = np.array([[1.0], [1.0]])
    
    # Function to compute LQR cost for a given gain K
    def compute_lqr_cost(K, A, B, Q, R, z0, n_steps=20):
        """Compute finite-horizon LQR cost for a given gain K."""
        cost = 0
        z = z0.copy()
        
        # Convert K from flat array to correct shape
        K_shaped = K.reshape(1, 2)
        
        for _ in range(n_steps):
            # Control input: u = -Kz
            u = -K_shaped @ z
            
            # Cost at current step
            cost += z.T @ (Q + K_shaped.T * R * K_shaped) @ z
            
            # Update state: z(t+1) = Az(t) + Bu(t)
            z = A @ z + B @ u
            
            # Check stability - return high cost if unstable
            if np.any(np.abs(z) > 1e6):
                return 1e10
                
        return float(cost)
    
    # Define high and low fidelity functions
    def f_high(K):
        """High-fidelity function (true system)."""
        cost = compute_lqr_cost(K, A_true, B_true, Q, R, z0)
        return np.log(cost)  # Log transform as in equation 17
    
    def f_low(K):
        """Low-fidelity function (approximate model)."""
        cost = compute_lqr_cost(K, A_approx, B_approx, Q, R, z0)
        return np.log(cost)  # Log transform as in equation 17
    
    # Create grid for controller gains
    x1_range = np.linspace(-0.5, 4.5, 26)  # x1 ∈ [-0.5, 4.5]
    x2_range = np.linspace(-3.5, 1.5, 26)  # x2 ∈ [-3.5, 1.5]
    
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack((X1.flatten(), X2.flatten()))
    
    # Create initial safe set (3 points known to be safe)
    # For demonstration purposes, we'll just pick 3 points we know are safe
    # In practice, these might come from prior knowledge or low-fidelity evaluation
    S0 = {
        (2.0, -1.0),
        (2.5, -1.5),
        (3.0, -2.0)
    }
    
    # Set safety threshold
    h = 0.0  # As specified in the paper
    
    # Create and run SafeSlope
    safeslope = SafeSlope(X_grid, h, S0, delta_f=0.1, delta_m=0.1)
    best_x_sf, best_y_sf, X_data_sf, y_data_sf = safeslope.optimize(f_high, num_iterations=150)
    
    # Create and run Multi-Fidelity SafeSlope
    mf_safeslope = MultiFidelitySafeSlope(X_grid, h, S0, rho=0.5, delta_f=0.1, delta_m=0.1)
    best_x_mf, best_y_mf, X_data_mf, y_data_mf = mf_safeslope.optimize(f_high, f_low, num_iterations=150)
    
    # Create and run SafeUCB (simplified implementation)
    # This is a simplified implementation for comparison purposes
    
    # Compare results
    print("\nResults:")
    print(f"SafeSlope best controller gains: K = {best_x_sf}, Cost = {np.exp(best_y_sf)}")
    print(f"Multi-Fidelity SafeSlope best controller gains: K = {best_x_mf}, Cost = {np.exp(best_y_mf)}")
    
    # Plot results - cumulative regret and unsafe samples
    plt.figure(figsize=(12, 5))
    
    # Plot cumulative regret
    plt.subplot(1, 2, 1)
    regret_sf = np.cumsum([y - best_y_sf for y in y_data_sf])
    regret_mf = np.cumsum([y - best_y_mf for y in y_data_mf])
    
    plt.plot(range(len(regret_sf)), regret_sf, label='SafeSlope')
    plt.plot(range(len(regret_mf)), regret_mf, label='MF-SafeSlope')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.title('Cumulative Regret Comparison')
    
    # Plot number of unsafe samples
    plt.subplot(1, 2, 2)
    unsafe_sf = [y > h for y in y_data_sf]
    unsafe_mf = [y > h for y in y_data_mf]
    
    cum_unsafe_sf = np.cumsum(unsafe_sf)
    cum_unsafe_mf = np.cumsum(unsafe_mf)
    
    plt.plot(range(len(cum_unsafe_sf)), cum_unsafe_sf, label='SafeSlope')
    plt.plot(range(len(cum_unsafe_mf)), cum_unsafe_mf, label='MF-SafeSlope')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Unsafe Samples')
    plt.legend()
    plt.title('Cumulative Unsafe Samples')
    
    plt.tight_layout()
    plt.show()
    
    # Return results for further analysis
    return {
        'safeslope': {
            'best_x': best_x_sf,
            'best_y': best_y_sf,
            'X_data': X_data_sf,
            'y_data': y_data_sf
        },
        'mf_safeslope': {
            'best_x': best_x_mf,
            'best_y': best_y_mf,
            'X_data': X_data_mf,
            'y_data': y_data_mf
        }
    }

if __name__ == "__main__":
    run_example()