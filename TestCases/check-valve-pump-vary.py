# Adapted from J. D. Nault's PhD thesis: http://hdl.handle.net/1807/80861
# Taken from Section 5.6.1, figure 5.3, page 88

import numpy as np

import PhysicalElements as pe
import RootFinding as rf
import Simulator as sim
import Controllers


reservoirs = res = (
    pe.Reservoir(100),
    pe.Reservoir(150)
)

junctions = jncts = (
    
)

pipes = (
    pe.Pipe((res[0], res[1]), 1000, .5, 0.015, .0, .62),
)

true_a, true_b = -120 / 300 * 5 / 8, 120 / 300**2
scale = [0, -.5, .5, 1]
pump_params = [(true_a * (1 + a_scale), true_b * (1 + b_scale)) for a_scale in scale for b_scale in scale]
possible_pumps = []
for (a, b) in pump_params:
    possible_pumps.append((pe.Pump(pipes[0], a, b, 2e-1, 1/300, .5, 0, 300),))

angle_table = np.pi / 180 * (16.1 + np.arange(0, 48, step=4))
kf = lambda angle: np.interp(angle, angle_table,
                             np.array([1e-12, .16, .28, .40, .49, .56, .62, .67, .71, .77, .84, .95]))
kd = lambda angle: np.interp(angle, angle_table,
                             np.array([1e-12, .23, .40, .49, .55, .58, .54, .49, .44, .38, .27, .09]))

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
    pe.CheckValve(pipes[0], 6, 1.3558 * .235, cv_diameter, [min_angle, max_angle], damping_map, drag_map, flow_map, head_map, np.pi / 180 * 60.1, 0),
)

outage_start_time = 30

input_guess = [300]

water_density = rho = 1000
accel_due_to_gravity = g = 9.81

times = []
flows, pump_speeds, cv_positions, cv_speeds, aug_states_s = [], [], [], [], []
pump_heads, cv_heads, control_inputs, power_loss_sets = [], [], [], []

true_index = 0 # TODO don't hardcode?

for i, pumps in enumerate(possible_pumps):

    initial_condition_finder = icf = rf.WDN_RootFinder(reservoirs, junctions, pipes, pumps, input_guess, check_valves=check_valves)
    initial_guess = [s.x0 for s in icf.states]
    
    controller_ramp_time = 60 * 2
    
    const_controller = Controllers.ConstantRampController(initial_guess[-1], 50, outage_start_time, controller_ramp_time) # TODO pick end better?
    
    const_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [const_controller], check_valves=check_valves)
    
    ic_result = const_simulator.simulate([-100, 0], initial_guess[:-1], method='RK45')
    
    ic = ic_result.y[:,-1]
    
    start_target_flow = ic[pipes[0].state_idx]
    end_target_flow = 0.01

    pi_flow_controller = controller = c = Controllers.PISetpointLogisticFlowController(initial_guess[-1], initial_guess[-1], outage_start_time, controller_ramp_time/4, 0.05, pipes[0],
                                                            start_target_flow, end_target_flow, prop_gain=200, int_gain=50)
    
    pi_flow_sim = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pi_flow_controller], check_valves=check_valves)
    
    sim_time = outage_start_time + controller_ramp_time + 120
    sim_time_range = [0, sim_time]
    
    no_control_result = const_simulator.simulate(sim_time_range, ic)
    pi_flow_result = pi_flow_sim.simulate(sim_time_range, np.concatenate([ic, np.zeros(1)]), method='RK45')
    
    sim_result = pi_flow_result
    
    t = sim_result.t
    flow, pump_speed, cv_pos, cv_speed, *aug_states = sim_result.y
    
    pump_head = pumps[0].head_gain(sim_result.y)
    cv_head = check_valves[0].head_loss(sim_result.y)
    
    control_action = controller.control_action_builder()
    control_input = np.array([control_action(ct,cx) for ct, cx in zip(pi_flow_result.t,pi_flow_result.y.T)])

    time_steps = t[1:]-t[:-1]
    pump_energy = rho * g * pump_head[:-1] * flow[:-1] * time_steps * (t[:-1] > 30)
    cumul_pump_energy = np.cumsum(pump_energy)
    
    energy_amounts = np.array([2.2, 3.3, 4.4]) * 3600 * 1000 # kWh -> J
    power_loss_indices = []
    energy_labels = []
    for energy_amount in energy_amounts:
        power_loss_indices.append(np.searchsorted(cumul_pump_energy, energy_amount))
        energy_labels.append(f"{energy_amount / 3600 / 1000:.2f} kWh")
    
    if i == true_index:
        true_t = t
        
        true_flow = flow
        true_pump_speed = pump_speed
        true_cv_pos = cv_pos
        true_cv_speed = cv_speed
        
        true_pump_head = pump_head
        true_cv_head = cv_head
        true_control_input = control_input
        true_power_loss_indices = power_loss_indices
    else:
        times.append(t)
        
        flows.append(flow)
        pump_speeds.append(pump_speed)
        cv_positions.append(cv_pos)
        cv_speeds.append(cv_speed)
        
        pump_heads.append(pump_head)
        cv_heads.append(cv_head)
        control_inputs.append(control_input)
        power_loss_sets.append(power_loss_indices)
    
t = true_t

int_flows, int_pump_speeds, int_cv_positions, int_cv_speeds, int_aug_states_s = [], [], [], [], []
int_pump_heads, int_cv_heads, int_control_inputs, int_power_loss_sets = [], [], [], []
for i, time in enumerate(times):
    int_flow = np.interp(t, time, flows[i])
    int_pump_speed = np.interp(t, time, pump_speeds[i])
    int_cv_pos = np.interp(t, time, cv_positions[i])
    int_cv_speed = np.interp(t, time, cv_speeds[i])
    
    int_pump_head = np.interp(t, time, pump_heads[i])
    int_cv_head = np.interp(t, time, cv_heads[i])
    int_control_input = np.interp(t, time, control_inputs[i])
    
    int_flows.append(int_flow)
    int_pump_speeds.append(int_pump_speed)
    int_cv_positions.append(int_cv_pos)
    int_cv_speeds.append(int_cv_speed)
    
    int_pump_heads.append(int_pump_head)
    int_cv_heads.append(int_cv_head)
    int_control_inputs.append(int_control_input)
    
### Plotting

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{siunitx}'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# TODO make SI use / instead of ^-1


plt.figure()
fig, ax = plt.subplots()
plt.plot(t, true_pump_head, label="True")
for time, pump_head in zip(times, pump_heads):
    plt.plot(time, pump_head, label="")
    
ax_ins = ax.inset_axes([.5, 0.5, .47, .47], xlim=(80, 90), ylim=(47.4, 47.8))

ax_ins.plot(t, true_pump_head)
for time, pump_head in zip(times, pump_heads):
    ax_ins.plot(time, pump_head)

ax.indicate_inset_zoom(ax_ins, edgecolor="black")
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
#plt.legend()

plt.figure()
fig, ax = plt.subplots()
ax.plot(t, true_flow, label="True")
for time, flow in zip(times, flows):
    ax.plot(time, flow, label="")
    
ax_ins = ax.inset_axes([.5, 0.5, .47, .47], xlim=(29.98, 30.2), ylim=(0.756, 0.7576))
#                       xticklabels=[30, 30.1, 30.2], yticklabels=[0.7565, 0.7575]) # Just relabels, doesn't renumber
ax_ins.plot(t, true_flow)
for time, flow in zip(times, flows):
    ax_ins.plot(time, flow)

ax.indicate_inset_zoom(ax_ins, edgecolor="black")
plt.ylabel(r"Flow (\( \si{ \meter^{3} \; \second^{-1}} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
#plt.legend() # breaks with inset?

plt.figure()
plt.plot(t, true_pump_speed, label="No error")
for time, pump_speed in zip(times, pump_speeds):
    plt.plot(time, pump_speed, label="")
plt.ylabel(r"Radial Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
for pump_head in int_pump_heads:
    plt.plot(t, true_pump_head - pump_head, label="")
plt.ylabel(r"Difference in Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
#plt.legend()

plt.figure()
for flow in int_flows:
    plt.plot(t, true_flow - flow, label="")
plt.ylabel(r"Difference in Flow (\( \si{ \meter^{3} \; \second^{-1}} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
#plt.legend()

plt.figure()
for pump_speed in int_pump_speeds:
    plt.plot(t, true_pump_speed - pump_speed, label="")
plt.ylabel(r"Difference in Radial Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
#plt.legend()

# cmap = plt.get_cmap("tab10")
# colours = [cmap(i+1) for i in range(len(power_loss_indices))]
# colours = ["black" for i in range(len(power_loss_indices))]
 
# plt.figure()
# plt_val = pump_head
# plt.plot(t, plt_val, c=colours[0])
# plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
# offset_vals = [(5,0),(2,1),(-20,1)]
# for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
#     plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
# plt.ylabel(r"Head (\( \si{\meter} \))")
# plt.xlabel(r"Time (\( \si{\second} \))")
# # plt.legend()

# plt.figure()
# plt_val = flow
# plt.plot(t, plt_val, c=colours[0])
# plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
# offset_vals = [(5,0),(5,0),(-20,.03)]
# for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
#     plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
# plt.ylabel(r"Flow (\( \si{ \meter^{3} \; \second^{-1}} \))") # TODO bad workaround
# plt.xlabel(r"Time (\( \si{\second} \))")
# #plt.legend()

# plt.figure()
# plt_val = pump_speed
# plt.plot(t, plt_val, c=colours[0])
# plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
# offset_vals = [(5,0),(5,0),(-20,5)]
# for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
#     plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
# plt.ylabel(r"Radial Speed (\( \si{\radian \per \second} \))")
# plt.xlabel(r"Time (\( \si{\second} \))")
# # plt.legend()

# plt.figure()
# plt_val = cv_pos * 180 / np.pi
# plt.plot(t, plt_val, c=colours[0])
# plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
# offset_vals = [(5,0),(5,0),(-20,2)]
# for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
#     plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
# plt.ylabel(r"Angle (\( \si{\degree} \))")
# plt.xlabel(r"Time (\( \si{\second} \))")
# # plt.legend()

# plt.figure()
# plt_val = cv_speed * 180 / np.pi
# plt.plot(t, plt_val, c=colours[0])
# plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
# offset_vals = [(0,0),(0,0),(0,0)]
# for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
#     plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
# plt.ylabel(r"Radial Speed (\( \si{\degree \per \second} \))")
# plt.xlabel(r"Time (\( \si{\second} \))")
# # plt.legend()

# plt.figure()
# plt_val = cv_head
# plt.plot(t, plt_val, c=colours[0])
# plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
# offset_vals = [(5,0),(5,0),(-20,.02)]
# for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
#     plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
# plt.ylabel(r"Head (\( \si{\meter} \))")
# plt.xlabel(r"Time (\( \si{\second} \))")
# # plt.legend()