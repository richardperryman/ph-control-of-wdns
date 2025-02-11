import numpy as np
import scipy.optimize as optimize

import PhysicalElements as pe

class WDN_RootFinder(pe.WDN_Model):
    def __init__(self, reservoirs, junctions, pipes, pumps, torque_initial_states, check_valves=None):
        super().__init__(reservoirs, junctions, pipes, pumps, check_valves=check_valves)
        
        self.torques = []
        for pump, u0 in zip(self.pumps, torque_initial_states):
            torque = pe.Torque(pump.pipe, pump.a, pump.b, pump.inertia, pump.dimensionless_flow, pump.diameter, pump.damping, u0)
            pump.torque = torque
            torque.pump = pump
            self.torques.append(torque)
            
        self.states.extend(self.torques)
        
        self.make_derivatives()
        
    def make_derivatives(self):
        super().make_derivatives()
        
        for pump in self.pumps:
            def get_control_state(t, x):
                return x[pump.torque.state_idx]
            
            derivative = pump.derivative_builder(pump.pipe, get_control_state)
            self.derivatives[pump.state_idx] = derivative
        
        for torque in self.torques:
            def get_control_state(t, x):
                return x[torque.state_idx]
            
            derivative = torque.derivative_builder(torque.pump.pipe, get_control_state)
            self.derivatives[torque.state_idx] = derivative

    def find_equilibrium(self, t=0, initial_guess=None, equilibrium_validator=None, noise_generator=None, tolerance=1e-3, verbosity=0):
        if initial_guess is None:
            initial_guess = np.array([state.x0 for state in self.states])
            
        if equilibrium_validator is None:
            equilibrium_validator = lambda *args: True
            
        if noise_generator is None:
            noise_rng = np.random.default_rng(seed=1729)
            def noise_function():
                n = len(self.states)
                noise_values = np.zeros(n)
                for i in range(n):
                    noise_values[i] = noise_rng.normal(0, initial_guess[i] / 10)
                return noise_values
            noise_generator = noise_function
            
        model = lambda x: self.get_model()(t, x)
        guess = initial_guess
        
        result = optimize.root(model, guess)
        while not equilibrium_validator(result.x) or (np.abs(model(result.x)) > tolerance).any():
            if verbosity > 0:
                print(result.x)
            guess = initial_guess + noise_generator()
            result = optimize.root(model, guess)
        
        self.result = result
        return result
    
    def find_equilibrium_constrained(self, t=0, initial_guess=None, constraints=None, bounds=None, weights=None, tolerance=1e-3, verbosity=0):
        if initial_guess is None:
            initial_guess = np.array([state.x0 for state in self.states])
            
        if weights is None:
            weights = np.array([1 for state in self.states])
            
        model = lambda x: np.linalg.norm(self.get_model()(t, x) * weights, 2)
        guess = initial_guess
        
        result = optimize.minimize(model, guess, bounds=bounds, constraints=constraints)
        self.result = result
        return result