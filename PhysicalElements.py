import numpy as np
import itertools

import HelperFunctions as hf

g = 9.81
water_density = 1000


class Reservoir():
    def __init__(self, head):
        self.head = head


class State():
    def __init__(self, x0, disturbance=None):
        self.x0 = x0
        if disturbance is None:
            def disturbance(t, x): return 0
        self.disturbance = disturbance


class Junction(State):
    def __init__(self, surface_area, conductance, h0, disturbance=None):
        if disturbance is None:
            def outflow_disturbance(t, x):
                return -self.outflow(x)
            disturbance = outflow_disturbance
            
        super().__init__(h0, disturbance)
        self.surface_area = surface_area
        self.conductance = conductance

    def outflow(self, state, opening=1):
        if self.conductance == 0:
            return 0
        head = state[self.state_idx]
        if head < 0:
            return 0
        return np.sqrt(head) / self.conductance * opening

    def derivative_builder(self, inflows, outflows):
        def derivative(t, x):
            net_inflow = sum(x[inflow.state_idx] for inflow in inflows)
            net_outflow = sum(x[outflow.state_idx] for outflow in outflows)
            net_flow = net_inflow - net_outflow + self.disturbance(t, x)
            return net_flow / self.surface_area
        return derivative
    
class FixedOutflowJunction(Junction):

    def outflow(self, state, opening=1):
        if self.conductance == 0:
            return 0
        head = self.x0
        if head < 0:
            return 0
        return np.sqrt(head) / self.conductance * opening


class Pipe(State):
    def __init__(self, connection, length, diameter, dw_friction, loss_coefficient, q0, disturbance=None):
        super().__init__(q0, disturbance)
        self.start, self.end = connection
        self.length = length
        self.diameter = diameter
        self.dw_friction = dw_friction
        self.loss_coefficient = loss_coefficient

    def friction(self, state):
        k = 8 / g / np.pi ** 2 * self.length * self.dw_friction / self.diameter ** 5 + \
            8 * self.loss_coefficient / g / np.pi ** 2 / self.diameter ** 4
        flow = state[self.state_idx]
        return k * flow * np.abs(flow)
    
    def area(self):
        return np.pi * self.diameter ** 2 / 4

    def derivative_builder(self, upstream, downstream, pumps, check_valves):
        def derivative(t, x):
            up = hf.get_head(upstream, x)
            down = hf.get_head(downstream, x)
            pump_gain = sum(pump.head_gain(x) for pump in pumps)
            friction = self.friction(x)
            check_valve_loss = sum(cv.head_loss(x) for cv in check_valves)
            net_head = up - down - friction + \
                pump_gain - check_valve_loss + self.disturbance(t, x)
            scaling = self.length / g / (np.pi * self.diameter ** 2 / 4)
            return net_head / scaling
        return derivative

class HW_Pipe(Pipe):
    def friction(self, state):
        C = 10.67
        n = 1.852
        flow = state[self.state_idx]
        k = C * self.length / self.dw_friction ** n / self.diameter ** 4.87 + \
            8 * self.loss_coefficient / g / np.pi ** 2 / self.diameter ** 4 * np.abs(flow) ** (2-n)
        return k * flow * np.abs(flow)

class FillingPipeFlow(Pipe):
    def __init__(self, connection, max_length, diameter, dw_friction, loss_coefficient, height_map, q0, disturbance=None):
        super().__init__(connection, max_length, diameter, dw_friction, loss_coefficient, q0, disturbance)
        self.height_map = height_map
        
    def get_area(self):
        return np.pi * (self.diameter / 2)**2
        
    def friction(self, state, length_state):
        column_length = state[length_state.state_idx]
        k = 8 / g / np.pi ** 2 * column_length * self.dw_friction / self.diameter ** 5 + \
            8 * self.loss_coefficient / g / np.pi ** 2 / self.diameter ** 4
        flow = state[self.state_idx]
        return k * flow * np.abs(flow)
        
    def derivative_builder(self, upstream, length_state, pumps, check_valves):
        def derivative(t, x):
            column_length = x[length_state.state_idx]
            up = hf.get_head(upstream, x)
            down = self.height_map(column_length)
            pump_gain = sum(pump.head_gain(x) for pump in pumps)
            friction = self.friction(x, length_state)
            check_valve_loss = sum(cv.head_loss(x) for cv in check_valves)
            net_head = up - down - friction + \
                pump_gain - check_valve_loss + self.disturbance(t, x)
            scaling = column_length / g / (np.pi * self.diameter ** 2 / 4)
            return net_head / scaling
        return derivative

class FillingPipeLength(Junction):
    def __init__(self, l0, disturbance=None):
        State.__init__(self, l0, disturbance)
        
    def derivative_builder(self, filling_pipes, _): # not a great fit
        def derivative(t, x):
            filling_pipe = filling_pipes[0]
            flow_rate = x[filling_pipe.state_idx]
            return flow_rate / filling_pipe.get_area()
        return derivative

class Pump(State):
    def __init__(self, pipe, a, b, inertia, dimensionless_flow, diameter, damping, w0, disturbance=None):
        super().__init__(w0, disturbance)
        self.pipe = pipe
        self.a = a
        self.b = b
        self.inertia = inertia
        self.dimensionless_flow = dimensionless_flow
        self.diameter = diameter
        self.damping = damping

    def linear_damping(self, state):
        return self.damping * state[self.state_idx]

    def pump_affinity_factor(self):
        return self.dimensionless_flow * self.diameter ** 3
    
    def head_gain(self, state):
        speed = state[self.state_idx]
        flow = state[self.pipe.state_idx]
        return (self.a * flow + self.b * speed) * speed

    def loss_torque(self, state):
        ratio = self.head_gain(state)
        return water_density * g * self.pump_affinity_factor() * ratio

    def derivative_builder(self, pipe, control_input):
        def derivative(t, x):
            loss_torque = self.loss_torque(x)
            damping = self.linear_damping(x)
            net_torque = - loss_torque - damping + \
                control_input(t, x) + self.disturbance(t, x)
            return net_torque / self.inertia
        return derivative


class Torque(Pump):
    def __init__(self, pipe, a, b, inertia, dimensionless_flow, diameter, damping, u0, disturbance=None):
        State.__init__(self, u0, disturbance)
        self.pipe = pipe
        self.a = a
        self.b = b
        self.inertia = inertia
        self.dimensionless_flow = dimensionless_flow
        self.diameter = diameter
        self.damping = damping

    def derivative_builder(self, pipe, control_input, scaling_factor=1e9):
        def derivative(t, x):
            loss_torque = self.loss_torque(x)
            damping = self.linear_damping(x)
            net_torque = - loss_torque - damping + \
                control_input(t, x) + self.disturbance(t, x)
            return - net_torque / scaling_factor
        return derivative

class CheckValvePosition(State):
    def __init__(self, angle_bounds, theta0, disturbance=None):
        State.__init__(self, theta0, disturbance)
        self.angle_min, self.angle_max = angle_bounds
        
    def derivative_builder(self, CV_Speed):
        def derivative(t, x):
            angle = x[self.state_idx]
            speed = x[CV_Speed.state_idx]
            above_max = angle > self.angle_max
            below_min = angle < self.angle_min
            if (above_max and speed > 0) or (below_min and speed < 0):
                return 0
            return speed
        return derivative

class CheckValveSpeed(State):
    def __init__(self, mass, inertia, diameter, angle_bounds, damping_map, drag_map, flow_map, w0, disturbance=None):
        State.__init__(self, w0, disturbance)
        self.mass = mass
        self.inertia = inertia
        self.diameter = diameter
        self.angle_min, self.angle_max = angle_bounds
        self.damping_map = damping_map
        self.drag_map = drag_map
        self.flow_map = flow_map
        
    def derivative_builder(self, CV_Position, pipe):
        def derivative(t, x):
            angle = x[CV_Position.state_idx]
            speed = x[self.state_idx]
            flow = x[pipe.state_idx]
            
            gravity = -g * self.mass * self.diameter/2 * np.sin(angle)
            damping = self.damping_map(angle) * speed ** 2 * (speed < 0)
            drag = -self.drag_map(angle) * speed * np.abs(speed)
            flow = self.flow_map(angle) * flow * np.abs(flow)
            net_torque = gravity + damping + drag + flow
            
            above_max = angle > self.angle_max
            below_min = angle < self.angle_min
            if above_max and net_torque > 0:
                return -100 * speed # TODO this is extremely hacky
            elif below_min and net_torque < 0:
                return -100 * speed
            
            return net_torque / self.inertia
        return derivative
    
class CheckValve():
    def __init__(self, pipe, mass, inertia, diameter, angle_bounds, damping_map, drag_map, flow_map, head_map, theta0, w0, pos_dist=None, angle_dist=None):
        self.Position = CheckValvePosition(angle_bounds, theta0, pos_dist)
        self.Speed = CheckValveSpeed(mass, inertia, diameter, angle_bounds, damping_map, drag_map, flow_map, w0, angle_dist)
        self.pipe = pipe
        self.head_map = head_map
        
        self.Position.CV_Speed = self.Speed
        self.Position.pipe = pipe
        
        self.Speed.CV_Position = self.Position
        self.Speed.pipe = pipe
        
    def head_loss(self, state):
        cv_angle = state[self.Position.state_idx]
        flow = state[self.pipe.state_idx]
        return self.head_map(cv_angle) * flow * np.abs(flow)

class WDN_Model():
    def __init__(self, reservoirs, junctions, pipes, pumps, check_valves=None):
        self.reservoirs = reservoirs
        self.junctions = junctions
        self.pipes = pipes
        self.pumps = pumps
        self.check_valves = check_valves or []
        
        self.states = list(itertools.chain(junctions, pipes, pumps))
        for check_valve in self.check_valves:
            self.states.append(check_valve.Position)
            self.states.append(check_valve.Speed)
        
    def make_derivatives(self):
        self.derivatives = [0 for state in self.states]
        sig = hf.state_index_generator()
        for state in self.states:
            state.state_idx = next(sig)
            
        for junction in self.junctions:
            junction.inflows = []
            junction.outflows = []
            for pipe in self.pipes:
                if pipe.start == junction:
                    junction.outflows.append(pipe)
                elif pipe.end == junction:
                    junction.inflows.append(pipe)
                    
            derivative = junction.derivative_builder(junction.inflows, junction.outflows)
            self.derivatives[junction.state_idx] = derivative
            
        for pipe in self.pipes:
            pipe.pumps = []
            for pump in self.pumps:
                if pump.pipe == pipe:
                    pipe.pumps.append(pump)
            pipe.check_valves = []
            for check_valve in self.check_valves:
                if check_valve.pipe == pipe:
                    pipe.check_valves.append(check_valve)
                    
            derivative = pipe.derivative_builder(pipe.start, pipe.end, pipe.pumps, pipe.check_valves)
            self.derivatives[pipe.state_idx] = derivative
            
        for check_vale in self.check_valves:
            derivative = check_valve.Position.derivative_builder(check_valve.Speed)
            self.derivatives[check_valve.Position.state_idx] = derivative
            
            derivative = check_valve.Speed.derivative_builder(check_valve.Position, check_valve.pipe)
            self.derivatives[check_valve.Speed.state_idx] = derivative
            
    def get_model(self):
        def model(t, x):
            n = len(self.derivatives)
            x_dot = np.zeros(n)
            for i in range(n):
                x_dot[i] = self.derivatives[i](t, x)
            return x_dot
        return model
    
    def check_derivatives(self, t=0, x=None):
        if x is None:
            x = np.array([state.x0 for state in self.states])
        return [derivative(t, x) for derivative in self.derivatives]