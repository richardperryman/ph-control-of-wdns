import numpy as np

import PhysicalElements as pe
import RootFinding as rf
import Simulator as sim
import Controllers
import Disturbances

reservoirs = res = (
    pe.Reservoir(0),
    pe.Reservoir(50),
)

# surface area, conductance, h0
junctions = jncts = (
    pe.Junction(.001,   0, 70.5),
    pe.Junction(.001, 200, 67.6),
    pe.Junction(.001, 300, 66.6),
    pe.Junction(.001, 100, 66.3), # worked with 200
    pe.Junction(.001, 300, 65.5),
    pe.Junction(.001, 300, 66.3),
    pe.Junction(.001, 240, 51.2),
)

fire_start_time = 100
fire_valve_opening_duration = 10
fire_disturbance = Disturbances.OutflowValveOpeningDisturbance(junctions[3], fire_start_time, fire_valve_opening_duration)
junctions[3].disturbance = fire_disturbance.disturbance_builder()

# (start, end), length (m), diameter (m), darcy-weisback friction factor, local loss coefficient, q0
pipes = (
    pe.Pipe((res[0], jncts[0]),   1000, .40, 0.015, .9, .183),
    pe.Pipe((jncts[0], jncts[1]), 1000, .40, 0.015, .9, .149),
    pe.Pipe((jncts[1], jncts[2]), 1000, .40, 0.015, .9, .083),
    pe.Pipe((jncts[0], jncts[3]), 1000, .20, 0.015, .9, .034),
    pe.Pipe((jncts[1], jncts[4]), 1000, .20, 0.015, .9, .024),
    pe.Pipe((jncts[2], jncts[5]), 1000, .40, 0.015, .9, .056),
    pe.Pipe((jncts[3], jncts[4]), 1000, .30, 0.015, .9, .034),
    pe.Pipe((jncts[5], jncts[4]), 1000, .25, 0.015, .9, .029),
    pe.Pipe((jncts[4], jncts[6]), 1000, .20, 0.015, .9, .060),
#    pe.Pipe((jncts[5], res[1]),   1000, .30, 0.015, .9, .0340),
    pe.Pipe((jncts[6], res[1]),   1000, .25, 0.015, .9, .030), # Commented out the wrong one...
)

# link id, a, b, inertia, Cq, diameter (m), damping, w0 (rad/s)
pumps = (
    pe.Pump(pipes[0], -120 / 300 * 5 / 8, 120 / 300**2, 2e-1, 1/300, .5, 0.1, 254.21),
)

input_guess = [330.167]

initial_condition_finder = icf = rf.WDN_RootFinder(reservoirs, junctions, pipes, pumps, input_guess)
initial_guess = [s.x0 for s in icf.states]

const_controller = Controllers.ConstantController(initial_guess[-1])
prop_pressure_controller = Controllers.PropPressureController(initial_guess[-1], jncts[0], initial_guess[jncts[0].state_idx], prop_gain=4)
pi_pressure_controller = Controllers.PIPressureController(initial_guess[-1], jncts[0], initial_guess[jncts[0].state_idx], prop_gain=4, int_gain=.5)
pd_pressure_controller = Controllers.PDPressureController(initial_guess[-1], jncts[0], initial_guess[jncts[0].state_idx], prop_gain=4, der_gain=50)

const_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [const_controller])
prop_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [prop_pressure_controller])
pi_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pi_pressure_controller])
pd_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pd_pressure_controller])

ic_result = const_simulator.simulate([0, 50], initial_guess[:-1])

ic = ic_result.y[:,-1]

no_control_result = const_simulator.simulate([0, 300], ic)
prop_pressure_result = prop_simulator.simulate([0, 300], ic)
pi_pressure_result = pi_simulator.simulate([0, 300], np.concatenate([ic, np.zeros(1)]))
pd_pressure_result = pd_simulator.simulate([0, 300], ic)

sim_result = no_control_result

### Plotting

t = sim_result.t
head, flow, speed, *aug = np.split(sim_result.y, np.cumsum([len(junctions), len(pipes), len(pumps)]))

const_obs_head = no_control_result.y[0,:]
prop_obs_head = prop_pressure_result.y[0,:]
pi_obs_head = pi_pressure_result.y[0,:]
pd_obs_head = pd_pressure_result.y[0,:]

const_fire_head = no_control_result.y[3,:]
prop_fire_head = prop_pressure_result.y[3,:]
pi_fire_head = pi_pressure_result.y[3,:]
pd_fire_head = pd_pressure_result.y[3,:]

obs_pipe_idx = pipes[0].state_idx
const_obs_flow = no_control_result.y[obs_pipe_idx,:]
prop_obs_flow = prop_pressure_result.y[obs_pipe_idx,:]
pi_obs_flow = pi_pressure_result.y[obs_pipe_idx,:]
pd_obs_flow = pd_pressure_result.y[obs_pipe_idx,:]

obs_speed_idx = pumps[0].state_idx
const_obs_speed = no_control_result.y[obs_speed_idx,:]
prop_obs_speed = prop_pressure_result.y[obs_speed_idx,:]
pi_obs_speed = pi_pressure_result.y[obs_speed_idx,:]
pd_obs_speed = pd_pressure_result.y[obs_speed_idx,:]

fire_head = head[3,:]

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{siunitx}'

plt.figure()
head_lines = plt.plot(t, head.T)
plt.legend(head_lines, range(3, len(head_lines) + 3))
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")


def node_idx(node): # TODO move to helper functions?
    if isinstance(node, pe.Reservoir):
        return reservoirs.index(node) + 1
    return node.state_idx + len(reservoirs) + 1
plt.figure()
flow_lines = plt.plot(t, flow.T)
plt.legend(flow_lines, [f"{node_idx(pipes[i].start)}-{node_idx(pipes[i].end)}" for i in range(len(flow_lines))])
plt.ylabel(r"Flow (\( \si{\meter^{3} /per /second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
speed_lines = plt.plot(t, speed.T)
plt.legend(speed_lines, range(len(speed_lines)))
plt.ylabel(r"Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_obs_head.T, label="Constant")
plt.plot(t, prop_obs_head.T, label="Proportional")
plt.plot(t, pi_obs_head.T, label="PI")
plt.plot(t, pd_obs_head.T, label="PD")
plt.legend()
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_fire_head.T, label="Constant")
plt.plot(t, prop_fire_head.T, label="Proportional")
plt.plot(t, pi_fire_head.T, label="PI")
plt.plot(t, pd_fire_head.T, label="PD")
plt.legend()
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_obs_flow.T, label="Constant")
plt.plot(t, prop_obs_flow.T, label="Proportional")
plt.plot(t, pi_obs_flow.T, label="PI")
plt.plot(t, pd_obs_flow.T, label="PD")
plt.legend()
plt.ylabel(r"Flow (\( \si{\meter^{3} /per /second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_obs_speed.T, label="Constant")
plt.plot(t, prop_obs_speed.T, label="Proportional")
plt.plot(t, pi_obs_speed.T, label="PI")
plt.plot(t, pd_obs_speed.T, label="PD")
plt.legend()
plt.ylabel(r"Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")








