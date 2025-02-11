import numpy as np

class Controller():
    def __init__(self):
        self.has_augmented_states = False
        
class StatefulController(Controller):
    def __init__(self):
        self.has_augmented_states = True
        
    def set_aug_state(self, state):
        self.aug_state = state
    
class ConstantController(Controller):
    def __init__(self, constant_input):
        super().__init__()
        self.constant_input = constant_input
        
    def control_action_builder(self):
        def control_action(t, x):
            return self.constant_input
        return control_action
    
class PISpeedController(StatefulController):
    def __init__(self, eqbm_input, pump, target_speed, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.eqbm_input = eqbm_input
        self.pump = pump
        self.target_speed = target_speed
        self.prop_gain = prop_gain
        self.int_gain = int_gain
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_pump_speed = x[self.pump.state_idx]
            return target_pump_speed - self.target_speed
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_speed = x[self.pump.state_idx]
            prop_action = self.prop_gain * (target_node_speed - self.target_speed)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            return self.eqbm_input - prop_action - int_action
        return control_action
    
class PIFlowController(StatefulController):
    def __init__(self, eqbm_input, pump, target_flow, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.eqbm_input = eqbm_input
        self.pump = pump
        self.target_flow = target_flow
        self.prop_gain = prop_gain
        self.int_gain = int_gain
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_pump_flow = x[self.pump.pipe.state_idx]
            return target_pump_flow - self.target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_flow = x[self.pump.pipe.state_idx]
            prop_action = self.prop_gain * (target_node_flow - self.target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            return self.eqbm_input - prop_action - int_action
        return control_action
    
class PropPressureController(Controller):
    def __init__(self, eqbm_input, target_node, target_pressure, prop_gain = 1):
        super().__init__()
        self.eqbm_input = eqbm_input
        self.target_node = target_node
        self.target_pressure = target_pressure
        self.prop_gain = prop_gain
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            prop_action = self.prop_gain * (target_node_pressure - self.target_pressure)
            return self.eqbm_input - prop_action
        return control_action

class PIPressureController(StatefulController):
    def __init__(self, eqbm_input, target_node, target_pressure, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.eqbm_input = eqbm_input
        self.target_node = target_node
        self.target_pressure = target_pressure
        self.prop_gain = prop_gain
        self.int_gain = int_gain
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            return target_node_pressure - self.target_pressure
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            prop_action = self.prop_gain * (target_node_pressure - self.target_pressure)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            return self.eqbm_input - prop_action - int_action
        return control_action

class PDPressureController(Controller):
    def __init__(self, eqbm_input, target_node, target_pressure, prop_gain = 1, der_gain = 1):
        super().__init__()
        self.eqbm_input = eqbm_input
        self.target_node = target_node
        self.target_pressure = target_pressure
        self.prop_gain = prop_gain
        self.der_gain = der_gain
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            prop_action = self.prop_gain * (target_node_pressure - self.target_pressure)
            target_node_derivative = self.target_node.derivative_builder(self.target_node.inflows, self.target_node.outflows)
            der_action = self.der_gain * target_node_derivative(t, x)
            return self.eqbm_input - prop_action - der_action
        return control_action

class PIDPressureController(StatefulController):
    def __init__(self, eqbm_input, target_node, target_pressure, prop_gain = 1, int_gain = 1, der_gain = 1):
        super().__init__()
        self.eqbm_input = eqbm_input
        self.target_node = target_node
        self.target_pressure = target_pressure
        self.prop_gain = prop_gain
        self.int_gain = int_gain
        self.der_gain = der_gain
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            return target_node_pressure - self.target_pressure
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            prop_action = self.prop_gain * (target_node_pressure - self.target_pressure)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            target_node_derivative = self.target_node.derivative_builder(self.target_node.inflows, self.target_node.outflows)
            der_action = self.der_gain * target_node_derivative(t, x)
            return self.eqbm_input - prop_action - int_action - der_action
        return control_action
    
class ConstantRampController(Controller):
    def __init__(self, constant_start, constant_end, ramp_start, ramp_time):
        super().__init__()
        self.constant_start = constant_start
        self.constant_end = constant_end
        self.ramp_start = ramp_start
        self.ramp_time = ramp_time
        
        self.ramp_end = self.ramp_start + self.ramp_time
        self.slope = (self.constant_start - self.constant_end) / self.ramp_time
        
    def control_action_builder(self):
        def control_action(t, x):
            if t < self.ramp_start:
                return self.constant_start
            elif t < self.ramp_end:
                effective_time = t - self.ramp_start
                return self.constant_start - self.slope * effective_time
            return self.constant_end
        return control_action
    
class PDRampPressureController(Controller):
    def __init__(self, start_eqbm, end_eqbm, ramp_start, ramp_time, target_node, start_target_pressure, end_target_pressure, prop_gain = 1, der_gain = 1):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.ramp_start = ramp_start
        self.ramp_time = ramp_time
        self.target_node = target_node
        self.start_target_pressure = start_target_pressure
        self.prop_gain = prop_gain
        self.der_gain = der_gain
        
        self.ramp_end = self.ramp_start + self.ramp_time
        self.slope = (self.start_eqbm - self.end_eqbm) / self.ramp_time
    
    def control_action_builder(self):
        def control_action(t, x):
            target_node_pressure = x[self.target_node.state_idx]
            prop_action = self.prop_gain * (target_node_pressure - self.start_target_pressure)
            target_node_derivative = self.target_node.derivative_builder(self.target_node.inflows, self.target_node.outflows)
            der_action = self.der_gain * target_node_derivative(t, x)
            
            if t < self.ramp_start:
                eqbm = self.start_eqbm
            elif t < self.ramp_end:
                effective_time = t - self.ramp_start
                eqbm = self.start_eqbm - self.slope * effective_time
            else:
                eqbm = self.end_eqbm
            
            return eqbm - prop_action - der_action
        return control_action
    
class PIDRampPressureController(StatefulController):
    def __init__(self, start_eqbm, end_eqbm, ramp_start, ramp_time, target_node, start_target_pressure, end_target_pressure, prop_gain = 1, int_gain = 1, der_gain = 1):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.ramp_start = ramp_start
        self.ramp_time = ramp_time
        self.target_node = target_node
        self.start_target_pressure = start_target_pressure
        self.end_target_pressure = end_target_pressure
        self.prop_gain = prop_gain
        self.int_gain = int_gain
        self.der_gain = der_gain
        
        self.ramp_end = self.ramp_start + self.ramp_time
        self.slope = (self.start_eqbm - self.end_eqbm) / self.ramp_time
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_pressure = self.start_target_pressure if t < self.ramp_start else self.end_target_pressure
            target_node_pressure = x[self.target_node.state_idx]
            return target_node_pressure - target_pressure
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_pressure = self.start_target_pressure if t < self.ramp_start else self.end_target_pressure
            target_node_pressure = x[self.target_node.state_idx]
            prop_action = self.prop_gain * (target_node_pressure - target_pressure)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            target_node_derivative = self.target_node.derivative_builder(self.target_node.inflows, self.target_node.outflows)
            der_action = self.der_gain * target_node_derivative(t, x)
            
            if t < self.ramp_start:
                eqbm = self.start_eqbm
            elif t < self.ramp_end:
                effective_time = t - self.ramp_start
                eqbm = self.start_eqbm - self.slope * effective_time
            else:
                eqbm = self.end_eqbm
            
            return eqbm - prop_action - int_action - der_action
        return control_action
    
class PIRampFlowController(StatefulController):
    def __init__(self, start_eqbm, end_eqbm, ramp_start, ramp_time, target_pipe, start_target_flow, end_target_flow, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.ramp_start = ramp_start
        self.ramp_time = ramp_time
        self.target_pipe = target_pipe
        self.start_target_flow = start_target_flow
        self.end_target_flow = end_target_flow
        self.prop_gain = prop_gain
        self.int_gain = int_gain
        
        self.ramp_end = self.ramp_start + self.ramp_time
        self.slope = (self.start_eqbm - self.end_eqbm) / self.ramp_time
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_flow = self.start_target_flow if t < self.ramp_start else self.end_target_flow
            target_node_flow = x[self.target_pipe.state_idx]
            return target_node_flow - target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_flow = self.start_target_flow if t < self.ramp_start else self.end_target_flow
            target_pipe_flow = x[self.target_pipe.state_idx]
            prop_action = self.prop_gain * (target_pipe_flow - target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            
            if t < self.ramp_start:
                eqbm = self.start_eqbm
            elif t < self.ramp_end:
                effective_time = t - self.ramp_start
                eqbm = self.start_eqbm - self.slope * effective_time
            else:
                eqbm = self.end_eqbm
            
            return eqbm - prop_action - int_action
        return control_action
    
class PILogisticFlowController(StatefulController):
    def __init__(self, start_eqbm, end_eqbm, start_time, inflection_time, slope_scale, target_pipe, start_target_flow, end_target_flow, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.start_time = start_time
        self.inflection_time = inflection_time
        self.slope_scale = slope_scale
        self.target_pipe = target_pipe
        self.start_target_flow = start_target_flow
        self.end_target_flow = end_target_flow
        self.prop_gain = prop_gain
        self.int_gain = int_gain
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_flow = self.start_target_flow if t < self.start_time else self.end_target_flow
            target_node_flow = x[self.target_pipe.state_idx]
            return target_node_flow - target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_flow = self.start_target_flow if t < self.start_time else self.end_target_flow
            target_pipe_flow = x[self.target_pipe.state_idx]
            prop_action = self.prop_gain * (target_pipe_flow - target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            
            if t < self.start_time:
                eqbm = self.start_eqbm
            else:
                effective_time = t - self.start_time
                height = self.end_eqbm - self.start_eqbm
                
                logistic_ramp = height / (1 + np.exp(-self.slope_scale * (effective_time - self.inflection_time)))
                
                eqbm = self.start_eqbm + logistic_ramp
            
            return eqbm - prop_action - int_action
        return control_action

class PISetpointRampFlowController(StatefulController):
    def __init__(self, start_eqbm, end_eqbm, ramp_start, ramp_time, target_pipe, start_target_flow, end_target_flow, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.ramp_start = ramp_start
        self.ramp_time = ramp_time
        self.target_pipe = target_pipe
        self.start_target_flow = start_target_flow
        self.end_target_flow = end_target_flow
        self.prop_gain = prop_gain
        self.int_gain = int_gain
        
        self.ramp_end = self.ramp_start + self.ramp_time
        self.slope = (self.start_target_flow - self.end_target_flow) / self.ramp_time
    
    def target_flow(self, t):
        if t < self.ramp_start:
            return self.start_target_flow
        elif t < self.ramp_end:
            effective_time = t - self.ramp_start
            return self.start_target_flow - self.slope * effective_time
        return  self.end_target_flow
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_flow = self.target_flow(t)
            target_node_flow = x[self.target_pipe.state_idx]
            return target_node_flow - target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_flow = self.target_flow(t)
            target_pipe_flow = x[self.target_pipe.state_idx]
            prop_action = self.prop_gain * (target_pipe_flow - target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            
            if t < self.ramp_start:
                eqbm = self.start_eqbm
            else:
                eqbm = self.end_eqbm
            
            return eqbm - prop_action - int_action
        return control_action
    
class PISetpointLogisticFlowController(StatefulController):
    def __init__(self, start_eqbm, end_eqbm, start_time, inflection_time, slope_scale, target_pipe, start_target_flow, end_target_flow, prop_gain = 1, int_gain = 1):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.start_time = start_time
        self.inflection_time = inflection_time
        self.slope_scale = slope_scale
        self.target_pipe = target_pipe
        self.start_target_flow = start_target_flow
        self.end_target_flow = end_target_flow
        self.prop_gain = prop_gain
        self.int_gain = int_gain
    
    def target_flow(self, t):
        if t < self.start_time:
            return self.start_target_flow
        effective_time = t - self.start_time
        height = self.end_target_flow - self.start_target_flow
        
        logistic_ramp = height / (1 + np.exp(-self.slope_scale * (effective_time - self.inflection_time)))
        
        return self.start_target_flow + logistic_ramp
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_flow = self.target_flow(t)
            target_node_flow = x[self.target_pipe.state_idx]
            return target_node_flow - target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_flow = self.target_flow(t)
            target_pipe_flow = x[self.target_pipe.state_idx]
            prop_action = self.prop_gain * (target_pipe_flow - target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            
            if t < self.start_time:
                eqbm = self.start_eqbm
            else:
                eqbm = self.end_eqbm
            
            return eqbm - prop_action - int_action
        return control_action
    
class PISetpointLogisticDiscreteUpdateFlowController(StatefulController):
    def __init__(self, start_eqbm, end_eqbm, start_time, inflection_time, slope_scale, target_pipe, start_target_flow, end_target_flow, prop_gain = 1, int_gain = 1, time_between_measurements = .01):
        super().__init__()
        self.start_eqbm = start_eqbm
        self.end_eqbm = end_eqbm
        self.start_time = start_time
        self.inflection_time = inflection_time
        self.slope_scale = slope_scale
        self.target_pipe = target_pipe
        self.start_target_flow = start_target_flow
        self.end_target_flow = end_target_flow
        self.prop_gain = prop_gain
        self.int_gain = int_gain
        self.time_between_measurements = time_between_measurements # not super trivial
    
    def target_flow(self, t):
        if t < self.start_time:
            return self.start_target_flow
        effective_time = t - self.start_time
        height = self.end_target_flow - self.start_target_flow
        
        logistic_ramp = height / (1 + np.exp(-self.slope_scale * (effective_time - self.inflection_time)))
        
        return self.start_target_flow + logistic_ramp
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            target_flow = self.target_flow(t)
            target_node_flow = x[self.target_pipe.state_idx]
            return target_node_flow - target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            target_flow = self.target_flow(t)
            target_pipe_flow = x[self.target_pipe.state_idx]
            prop_action = self.prop_gain * (target_pipe_flow - target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            
            if t < self.start_time:
                eqbm = self.start_eqbm
            else:
                eqbm = self.end_eqbm
            
            return eqbm - prop_action - int_action
        return control_action

class PIMultistageFlowController(StatefulController): # how to reset aug state?
    def __init__(self, eqbm_inputs, pump, target_flows, times, prop_gains, int_gains):
        super().__init__()
        self.eqbm_inputs = eqbm_inputs
        self.pump = pump
        self.target_flows = target_flows
        self.times = np.array(times)
        self.prop_gains = prop_gains
        self.int_gains = int_gains
        
    def update_time(self, t):
        controller_index = np.searchsorted(self.times, t)
        
        self.eqbm_input = self.eqbm_inputs[controller_index]
        self.target_flow = self.target_flows[controller_index]
        self.prop_gain = self.prop_gains[controller_index]
        self.int_gain = self.int_gains[controller_index]
    
    def derivative_builder(self):
        def aug_derivative(t, x):
            self.update_time(t)
            target_pump_flow = x[self.pump.pipe.state_idx]
            return target_pump_flow - self.target_flow
        return aug_derivative
    
    def control_action_builder(self):
        def control_action(t, x):
            self.update_time(t)
            target_node_flow = x[self.pump.pipe.state_idx]
            prop_action = self.prop_gain * (target_node_flow - self.target_flow)
            int_action = self.int_gain * x[self.aug_state.state_idx]
            return self.eqbm_input - prop_action - int_action
        return control_action