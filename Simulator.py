#import numpy as np
import scipy.integrate as integrate

import PhysicalElements as pe

class ControllerState(pe.State):
    def __init__(self, x0):
        super().__init__(x0)

class WDN_Simulator(pe.WDN_Model):
    def __init__(self, reservoirs, junctions, pipes, pumps, controllers, check_valves=None):
        super().__init__(reservoirs, junctions, pipes, pumps, check_valves=check_valves)
        
        self.controllers = controllers
        for controller in controllers:
            if controller.has_augmented_states:
                aug_state = ControllerState(0)
                controller.set_aug_state(aug_state)
                self.states.append(aug_state)
        
        self.make_derivatives()
        
    def make_derivatives(self):
        super().make_derivatives()
        
        for pump, controller in zip(self.pumps, self.controllers):
            derivative = pump.derivative_builder(pump.pipe, controller.control_action_builder())
            self.derivatives[pump.state_idx] = derivative
        
        for controller in self.controllers:
            if controller.has_augmented_states:
                derivative = controller.derivative_builder()
                self.derivatives[controller.aug_state.state_idx] = derivative
    
    def make_events(self):
        events = []
        def negative_state_event_builder(state):
            def event(t, x):
                return x[state.state_idx]
            event.terminal = True
            event.state = state
            return event
            
        for junction in self.junctions:
            event = negative_state_event_builder(junction)
            events.append(event)
            
        for pump in self.pumps:
            event = negative_state_event_builder(pump)
            events.append(event)
            
            event = negative_state_event_builder(pump.pipe)
            events.append(event)
            
        return events
    
    def simulate(self, time_range, initial_state, add_default_events=True, events=None, max_step=0.01, method='RK45'):
        model = self.get_model()
        if events is None:
            events = []
        if add_default_events:
            events.extend(self.make_events())
        
        self.events = events
        result = integrate.solve_ivp(model, time_range, initial_state, max_step=max_step, events=events, method=method)
        self.result = result
        return result