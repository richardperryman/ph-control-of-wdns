import numpy as np

import PhysicalElements as pe
import RootFinding as rf
import Simulator as sim
import Controllers


reservoirs = res = (
    pe.Reservoir(0),
    pe.Reservoir(50)
)

esa = 1e-3

junctions = jncts = (
    pe.Junction(esa,   0, 65.0),
    pe.Junction(esa, 200, 58.0),
    pe.Junction(esa, 300, 54.7),
    pe.Junction(esa, 100, 59.3),
    pe.Junction(esa, 300, 53.5),
    pe.Junction(esa, 300, 52.2),
    pe.Junction(esa, 240, 50.1),
)

pipes = (
    pe.Pipe((res[0], jncts[0]),   1000, .40, 0.015, .9, .275),
    pe.Pipe((jncts[0], jncts[1]), 1000, .40, 0.015, .9, .236),
    pe.Pipe((jncts[1], jncts[2]), 1000, .40, 0.015, .9, .164),
    pe.Pipe((jncts[0], jncts[3]), 1000, .20, 0.015, .9, .038),
    pe.Pipe((jncts[1], jncts[4]), 1000, .20, 0.015, .9, .034),
    pe.Pipe((jncts[2], jncts[5]), 1000, .40, 0.015, .9, .140),
    pe.Pipe((jncts[3], jncts[4]), 1000, .20, 0.015, .9, .038),
    pe.Pipe((jncts[4], jncts[5]), 1000, .20, 0.015, .9, .018),
    pe.Pipe((jncts[4], jncts[6]), 1000, .20, 0.015, .9, .030),
    pe.Pipe((jncts[5], res[1]),   1000, .40, 0.015, .9, .134),
)


pumps = (
    pe.Pump(pipes[0], -120 / 300 * 5 / 8, 120 / 300**2, 2e-1, 1/300, .5, 0.1, 263.25),
)

angle_table = np.pi / 180 * (16.1 + np.arange(0, 48, step=4))
kf = lambda angle: np.interp(angle, angle_table,
                             np.array([1e-12, .16, .28, .40, .49, .56, .62, .67, .71, .77, .84, .95]))
kd = lambda angle: np.interp(angle, angle_table,
                             np.array([1e-12, .23, .40, .49, .55, .58, .54, .49, .44, .38, .27, .09]))

cv_diameter = pipes[0].diameter
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

input_guess = [330]

initial_condition_finder = icf = rf.WDN_RootFinder(reservoirs, junctions, pipes, pumps, input_guess, check_valves=check_valves)
initial_guess = [s.x0 for s in icf.states]

controller_ramp_time = 60 * 2

const_controller = Controllers.ConstantRampController(initial_guess[-1], 50, outage_start_time, controller_ramp_time) # TODO pick end better?

const_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [const_controller], check_valves=check_valves)

ic_result = const_simulator.simulate([-100, 0], initial_guess[:-1], method='RK45')

ic = ic_result.y[:,-1]

start_target_flow = ic[pipes[0].state_idx]
end_target_flow = 0.01

pi_flow_controller = controller = c = Controllers.PISetpointLogisticFlowController(initial_guess[-1], initial_guess[-1], outage_start_time, controller_ramp_time/4, 0.045, pipes[0],
                                                        start_target_flow, end_target_flow, prop_gain=200, int_gain=50)

pi_flow_sim = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pi_flow_controller], check_valves=check_valves)

sim_time = outage_start_time + controller_ramp_time + 120
sim_time_range = [0, sim_time]

pi_flow_result = pi_flow_sim.simulate(sim_time_range, np.concatenate([ic, np.zeros(1)]), method='RK45')

sim_result = pi_flow_result

t = sim_result.t
head, flow, pump_speed, (cv_pos, cv_speed), *aug_states = np.split(sim_result.y, np.cumsum([len(jncts), len(pipes), len(pumps), 2 * len(check_valves)]))

pump_head = pumps[0].head_gain(sim_result.y)
cv_head = check_valves[0].head_loss(sim_result.y)

pump_flow = flow[0]

control_action = controller.control_action_builder()
control_input = np.array([control_action(ct,cx) for ct, cx in zip(pi_flow_result.t,pi_flow_result.y.T)])

water_density = rho = 1000
accel_due_to_gravity = g = 9.81
time_steps = t[1:]-t[:-1]
pump_energy = rho * g * pump_head[:-1] * pump_flow[-1] * time_steps * (t[:-1] > 30)
cumul_pump_energy = np.cumsum(pump_energy)

energy_amounts = np.array([.01, .05, .1]) * 3600 * 1000 # kWh -> J
power_loss_indices = []
energy_labels = []
for energy_amount in energy_amounts:
    power_loss_indices.append(np.searchsorted(cumul_pump_energy, energy_amount))
    energy_labels.append(f"{energy_amount / 3600 / 1000:.2f} kWh")
    
### Plotting

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = ('\\usepackage{amsmath}\n\\usepackage{siunitx}\n\\sisetup{per-mode=power,inter-unit-product=\;}' + 
                                       '\\usepackage{newtxtext}\n\\usepackage{newtxmath}')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
# TODO DPI
# TODO make SI use / instead of ^-1

cmap = plt.get_cmap("tab10")
colours = [cmap(i+1) for i in range(len(power_loss_indices))]
colours = ["black" for i in range(len(power_loss_indices))]
 
plt.figure()
plt_val = pump_head
plt.plot(t, plt_val, c=colours[0])
plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
offset_vals = [(5,0),(2,1),(5,-1)]
for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
    plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
# plt.legend()

plt.figure()
plt_val = pump_flow
plt.plot(t, plt_val, c=colours[0])
plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
offset_vals = [(5,0),(5,0),(5,0)]
for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
    plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
plt.ylabel(r"Flow (\( \si[per-mode=power,inter-unit-product=\;]{\meter^{3} \; \second}^{-1} \))") # TODO bad workaround
plt.xlabel(r"Time (\( \si{\second} \))")
#plt.legend()

plt.figure()
plt_val = pump_speed[0]
plt.plot(t, plt_val, c=colours[0])
plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
offset_vals = [(5,0),(5,0),(5,-2)]
for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
    plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
plt.ylabel(r"Radial Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
# plt.legend()

plt.figure()
plt_val = cv_pos * 180 / np.pi
plt.plot(t, plt_val, c=colours[0])
plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
offset_vals = [(5,0),(5,0),(5,0)]
for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
    plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
plt.ylabel(r"Angle (\( \si{\degree} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
# plt.legend()

plt.figure()
plt_val = cv_speed * 180 / np.pi
plt.plot(t, plt_val, c=colours[0])
plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
offset_vals = [(5,0),(-40,0),(5,0)]
for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
    plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
plt.ylabel(r"Radial Speed (\( \si{\degree \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
# plt.legend()

plt.figure()
plt_val = cv_head
plt.plot(t, plt_val, c=colours[0])
plt.scatter(t[power_loss_indices], plt_val[power_loss_indices], c=colours)
offset_vals = [(5,0),(5,0),(-40,.0)]
for i, (power_loss_index, energy_label, (x_off, y_off)) in enumerate(zip(power_loss_indices, energy_labels, offset_vals)):
    plt.annotate(energy_label, (t[power_loss_index], plt_val[power_loss_index]), (t[power_loss_index] + x_off, plt_val[power_loss_index] + y_off))
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
# plt.legend()