# Adapted from J. D. Nault's PhD thesis: http://hdl.handle.net/1807/80861
# Taken from Section 5.6.1, figure 5.3, page 88

import numpy as np

import PhysicalElements as pe
import Simulator as sim
import Controllers

reservoirs = res = (
    pe.Reservoir(0),
)

junctions = jncts = (
    pe.FillingPipeLength(5),
)

pipe_length = 1000
pipe_final_height = 150
def linear_ramp(length):
    return (length / pipe_length) * pipe_final_height
linear_ramp.name = "Linear"
def logistic_ramp(length, inflection_point_ratio = 0.5, time_scale = .02):
    inflection_distance = pipe_length * inflection_point_ratio
    return pipe_final_height / (1 + np.exp(-time_scale * (length - inflection_distance)))
logistic_ramp.name = "Logistic"
early_logistic_ramp = lambda length:logistic_ramp(length, inflection_point_ratio=.25)
early_logistic_ramp.name = "Early Logistic"
late_logistic_ramp = lambda length:logistic_ramp(length, inflection_point_ratio=.75)
late_logistic_ramp.name = "Late Logistic"
two_stage_logistic = lambda length: (logistic_ramp(length, inflection_point_ratio=.25) + logistic_ramp(length, inflection_point_ratio=.75)) / 2
two_stage_logistic.name = "Two Stage Logistic"

pipe_profile = lambda x: linear_ramp(x)

pipes = (
    pe.FillingPipeFlow((res[0], jncts[0]), 1000, .5, 0.015, .0, pipe_profile, 0*.001),
)

pumps = (
    pe.Pump(pipes[0], -120 / 300 * 5 / 8, 120 / 300**2, 2e-1, 1/300, .5, 0, 0*30),
)

angle_table = np.pi / 180 * (16.1 + np.arange(0, 48, step=4))
kf = lambda angle: np.interp(angle, angle_table,
                             np.array([1e-3, .16, .28, .40, .49, .56, .62, .67, .71, .77, .84, .95]))
kd = lambda angle: np.interp(angle, angle_table,
                             np.array([1e-3, .23, .40, .49, .55, .58, .54, .49, .44, .38, .27, .09]))

cv_diameter = .5
cv_area = np.pi * (cv_diameter / 2) ** 2
min_angle = np.pi * 16.1 / 180
max_angle = np.pi * (16.1 + 44) / 180
B, C, F, G = 0, 0, 25, 50

damping_map = lambda angle: 0
drag_map = lambda angle: ((C + G) / kd(angle)) ** 2
flow_map = lambda angle: (cv_area * (B + F) / kf(angle)) ** 2
head_map = lambda angle: (1/(kf(angle) - 0)) ** 2
check_valves = (
    pe.CheckValve(pipes[0], 6, 1.3558 * .235, cv_diameter, [min_angle, max_angle], damping_map, drag_map, flow_map, head_map, np.pi / 180 * 16.1, 0),
)

target_flow = 0.2 * pipes[0].get_area()

controller_ramp_time = 60 * 2
equilibrium_input = 0 # 55 - 1.1554 * 50 # TODO CHECK?

pi_flow_controller = controller = Controllers.PIFlowController(equilibrium_input, pumps[0], target_flow, prop_gain=300, int_gain=100)

pi_flow_simulator = simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pi_flow_controller], check_valves=check_valves)

sim_time = 10000
sim_time_range = [0, sim_time]
def end_event(t, x):
    length = x[jncts[0].state_idx]
    return pipe_length - length
end_event.terminal = True

ic = [state.x0 for state in pi_flow_simulator.states] # TODO check

results = []
height_maps = [linear_ramp, early_logistic_ramp, logistic_ramp, late_logistic_ramp]
for height_map in height_maps:
    pipes[0].height_map = height_map
    
    pi_flow_result = pi_flow_simulator.simulate(sim_time_range, ic, add_default_events=False, max_step=0.1, events=[end_event], method='Radau')

    results.append(pi_flow_result)


### Plotting

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{siunitx}'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# TODO make SI use / instead of ^-1

lengths, flows, pump_speeds, cv_positions, cv_speeds = [], [], [], [], []
velocities = []
pump_heads = []
cv_heads = []
control_inputs = []
times = []
for sim_result in results:
    t = sim_result.t
    times.append(t)
    
    length, flow, pump_speed, cv_pos, cv_speed, *aug_states = sim_result.y
    lengths.append(length)
    flows.append(flow)
    velocities.append(flow / pipes[0].get_area())
    pump_speeds.append(pump_speed)
    cv_positions.append(cv_pos)
    cv_speeds.append(cv_speed)
    
    pump_head = np.array([pumps[0].head_gain(x) for x in sim_result.y.T])
    pump_heads.append(pump_head)
    
    cv_head = np.array([check_valves[0].head_loss(x) for x in sim_result.y.T])
    cv_heads.append(cv_head)
    
    control_action = controller.control_action_builder()
    control_input = np.array([control_action(0, x) for x in sim_result.y.T])
    control_inputs.append(control_input)

split_t = 25
symbols = ["-k", "--k", ":k", "-.k", "k"]

plt.figure()
for t, length, height_map, symbol in zip(times, lengths, height_maps, symbols):
    plt.plot(t[t<split_t], length[t<split_t], symbol, label=height_map.name)
plt.ylabel(r"Column Length (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, length, height_map, symbol in zip(times, lengths, height_maps, symbols):
    plt.plot(t[t>=split_t], length[t>=split_t], symbol, label=height_map.name)
plt.ylabel(r"Column Length (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, pump_head, height_map, symbol in zip(times, pump_heads, height_maps, symbols):
    plt.plot(t[t<split_t], pump_head[t<split_t], symbol, label=height_map.name)
plt.ylabel(r"Pump Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, pump_head, height_map, symbol in zip(times, pump_heads, height_maps, symbols):
    plt.plot(t[t>=split_t], pump_head[t>=split_t], symbol, label=height_map.name)
plt.ylabel(r"Pump Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, control_input, height_map in zip(times, control_inputs, height_maps):
    plt.plot(t, control_input, label=height_map.name)
plt.ylabel(r"Control input (\( \si{\newton \meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, velocity, height_map, symbol in zip(times, velocities, height_maps, symbols):
    plt.plot(t[t<split_t], velocity[t<split_t], symbol, label=height_map.name)
plt.ylabel(r"Flow Speed (\( \si{ \meter \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, velocity, height_map, symbol in zip(times, velocities, height_maps, symbols):
    plt.plot(t[t>=split_t], velocity[t>=split_t], symbol, label=height_map.name)
plt.ylabel(r"Flow speed (\( \si{ \meter \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, pump_speed, height_map, symbol in zip(times, pump_speeds, height_maps, symbols):
    plt.plot(t[t<split_t], pump_speed[t<split_t], symbol, label=height_map.name)
plt.ylabel(r"Pump Radial Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, pump_speed, height_map, symbol in zip(times, pump_speeds, height_maps, symbols):
    plt.plot(t[t>=split_t], pump_speed[t>=split_t], symbol, label=height_map.name)
plt.ylabel(r"Pump Radial Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, cv_pos, height_map, symbol in zip(times, cv_positions, height_maps, symbols):
    plt.plot(t[t<split_t], cv_pos[t<split_t] * 180 / np.pi, symbol, label=height_map.name)
plt.ylabel(r"Angle (\( \si{\degree} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, cv_pos, height_map, symbol in zip(times, cv_positions, height_maps, symbols):
    plt.plot(t[t>=split_t], cv_pos[t>=split_t] * 180 / np.pi, symbol, label=height_map.name)
plt.ylabel(r"Angle (\( \si{\degree} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, cv_speed, height_map in zip(times, cv_speeds, height_maps):
    plt.plot(t, cv_speed * 180 / np.pi, label=height_map.name)
plt.ylabel(r"Valve Radial Speed (\( \si{\degree \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, cv_head, height_map, symbol in zip(times, cv_heads, height_maps, symbols):
    plt.plot(t[t<split_t], cv_head[t<split_t], symbol, label=height_map.name)
plt.ylabel(r"Valve Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for t, cv_head, height_map, symbol in zip(times, cv_heads, height_maps, symbols):
    plt.plot(t[t>=split_t], cv_head[t>=split_t], symbol, label=height_map.name)
plt.ylabel(r"Valve Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
pipe_length_plot = np.arange(0, pipe_length, .1)
for height_map, symbol in zip(height_maps, symbols):
    plt.plot(pipe_length_plot, height_map(pipe_length_plot), symbol, label=height_map.name)
plt.ylabel(r"Pipe Elevation (\( \si{\meter} \))")
plt.xlabel(r"Pipe Length (\( \si{\meter} \))")
plt.legend()