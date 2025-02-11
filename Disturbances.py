import numpy as np

class Disturbance():
    ...
    
class ShutoffValveDisturbance(Disturbance):
    def __init__(self, pipe, valve_loss_coefficient, start_time, duration, initial_opening=1, min_opening=1e-6):
        self.pipe = pipe
        self.loss_coeff = valve_loss_coefficient
        self.start_time = start_time
        self.duration = duration
        self.initial_opening = initial_opening
        self.min_opening = min_opening
        
        self.end_time = self.start_time + self.duration
        

    def disturbance_builder(self):
        def valve_closing_disturbance(t, x):
            opening = self.initial_opening
            if t > self.start_time:
                opening = self.initial_opening * (1 - (t - self.start_time) / self.duration)
                opening = max(opening, self.min_opening)
            flow = x[self.pipe.state_idx]
            head_loss = self.loss_coeff / opening ** 2 * flow * np.abs(flow)
            return -head_loss
        return valve_closing_disturbance

class OutflowValveOpeningDisturbance(Disturbance):
    def __init__(self, junction, start_time, duration, initial_opening=0, max_opening=1):
        self.junction = junction
        self.start_time = start_time
        self.duration = duration
        self.initial_opening = initial_opening
        self.max_opening = max_opening
        
        self.end_time = self.start_time + self.duration
        

    def disturbance_builder(self):
        def valve_closing_disturbance(t, x):
            opening = self.initial_opening
            if t > self.start_time:
                slope = self.max_opening - self.initial_opening
                opening = slope * (t - self.start_time) / self.duration
                opening = min(opening, self.max_opening)
            
            return -self.junction.outflow(x, opening=opening)
        return valve_closing_disturbance
    
class OutflowValveTemporaryOpeningDisturbance(Disturbance):
    def __init__(self, junction, open_time, close_time, duration, initial_opening=0, max_opening=1):
        self.junction = junction
        self.open_time = open_time
        self.close_time = close_time
        self.duration = duration
        self.initial_opening = initial_opening
        self.max_opening = max_opening
        
        self.open_end_time = self.open_time + self.duration
        self.close_end_time = self.close_time + self.duration
        

    def disturbance_builder(self):
        def valve_closing_disturbance(t, x):
            opening = self.initial_opening
            if self.close_time > t > self.open_time:
                slope = self.max_opening - self.initial_opening
                opening = slope * (t - self.open_time) / self.duration
                opening = min(opening, self.max_opening)
            elif t >= self.close_time:
                slope = self.max_opening - self.initial_opening
                opening = self.max_opening - slope * (t - self.close_time) / self.duration
                opening = max(opening, self.initial_opening)
            
            return -self.junction.outflow(x, opening=opening)
        return valve_closing_disturbance

class OutageDisturbance(Disturbance):
    def __init__(self, pipe, start_time, closing_time, conductance=10, initial_opening=1, min_opening=0.01):
        self.pipe = pipe
        self.start_time = start_time
        self.closing_time = closing_time
        self.conductance = conductance
        self.initial_opening = initial_opening
        self.min_opening = min_opening
        
        self.end_time = self.start_time + self.closing_time
        
    def valve_head_loss(self, x, opening):
        flow = x[self.pipe.state_idx]
        # could be opening ** 2?
        return self.conductance / opening * flow * np.abs(flow)

    def disturbance_builder(self):
        def valve_closing_disturbance(t, x):
            opening = self.initial_opening
            if self.start_time < t <= self.end_time:
                slope = 0 - self.initial_opening
                opening = slope * (t - self.start_time) / self.closing_time
                opening = max(opening, self.min_opening)
            elif t > self.end_time:
                opening = self.min_opening # unsure, could be different, also unneeded?
            
            return -self.valve_head_loss(x, opening)
        return valve_closing_disturbance