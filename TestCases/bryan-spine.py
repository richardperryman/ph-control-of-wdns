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
    pe.Junction(.001,   0, 65.0),
    pe.Junction(.001, 200, 58.0),
    pe.Junction(.001, 300, 54.7),
    pe.Junction(.001, 100, 59.3),
    pe.Junction(.001, 300, 53.5),
    pe.Junction(.001, 300, 52.2),
    pe.Junction(.001, 240, 50.1),
)

fire_start_time = 30
fire_end_time = 100
fire_valve_opening_duration = 10
fire_disturbance = Disturbances.OutflowValveTemporaryOpeningDisturbance(junctions[3], fire_start_time, fire_end_time, fire_valve_opening_duration)
junctions[3].disturbance = fire_disturbance.disturbance_builder()

# (start, end), length (m), diameter (m), darcy-weisbach friction factor, local loss coefficient, q0
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

# link id, a, b, inertia, Cq, diameter (m), damping, w0 (rad/s)
pumps = (
    pe.Pump(pipes[0], -120 / 300 * 5 / 8, 120 / 300**2, 2e-1, 1/300, .5, 0.1, 263.25),
)

input_guess = [330.167]

initial_condition_finder = icf = rf.WDN_RootFinder(reservoirs, junctions, pipes, pumps, input_guess)
initial_guess = [s.x0 for s in icf.states]

target_pressure = initial_guess[jncts[0].state_idx]
target_speed = initial_guess[pumps[0].state_idx]
target_flow = initial_guess[pipes[0].state_idx]

const_controller = Controllers.ConstantController(initial_guess[-1])
#prop_pressure_controller = Controllers.PropPressureController(initial_guess[-1], jncts[0], target_pressure, prop_gain=4)
#pi_pressure_controller = Controllers.PIPressureController(initial_guess[-1], jncts[0], target_pressure, prop_gain=4, int_gain=.5)
pd_pressure_controller = Controllers.PDPressureController(initial_guess[-1], jncts[0], target_pressure, prop_gain=4, der_gain=50)
pid_pressure_controller = Controllers.PIDPressureController(initial_guess[-1], jncts[0], target_pressure, prop_gain=4, int_gain=.5, der_gain=50)
pi_speed_controller = Controllers.PISpeedController(initial_guess[-1], pumps[0], target_speed, prop_gain=.04, int_gain=.005)
pi_flow_controller = Controllers.PIFlowController(initial_guess[-1], pumps[0], target_flow, prop_gain=40, int_gain=5)

const_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [const_controller])
pd_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pd_pressure_controller])
pid_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pid_pressure_controller])
pi_speed_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pi_speed_controller])
pi_flow_simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [pi_flow_controller])

ic_result = const_simulator.simulate([-50, fire_start_time - 1], initial_guess[:-1])

ic = ic_result.y[:,-1]

sim_time = fire_start_time + fire_end_time + 100
sim_time_range = [0, sim_time]

no_control_result = const_simulator.simulate(sim_time_range, ic)
pd_pressure_result = pd_simulator.simulate(sim_time_range, ic)
pid_pressure_result = pid_simulator.simulate(sim_time_range, np.concatenate([ic, np.zeros(1)]))
pi_speed_result = pi_speed_simulator.simulate(sim_time_range, np.concatenate([ic, np.zeros(1)]))
pi_flow_result = pi_flow_simulator.simulate(sim_time_range, np.concatenate([ic, np.zeros(1)]))

sim_result = no_control_result

### Plotting

t = sim_result.t
head, flow, speed, *aug = np.split(sim_result.y, np.cumsum([len(junctions), len(pipes), len(pumps)]))

const_obs_head = no_control_result.y[0,:]
pd_obs_head = pd_pressure_result.y[0,:]
pid_obs_head = pid_pressure_result.y[0,:]
pi_speed_obs_head = pi_speed_result.y[0,:]
pi_flow_obs_head = pi_flow_result.y[0,:]

const_fire_head = no_control_result.y[3,:]
pd_fire_head = pd_pressure_result.y[3,:]
pid_fire_head = pid_pressure_result.y[3,:]
pi_speed_fire_head = pi_speed_result.y[3,:]
pi_flow_fire_head = pi_flow_result.y[3,:]

obs_pipe_idx = pipes[0].state_idx
const_obs_flow = no_control_result.y[obs_pipe_idx,:]
pd_obs_flow = pd_pressure_result.y[obs_pipe_idx,:]
pid_obs_flow = pid_pressure_result.y[obs_pipe_idx,:]
pi_speed_obs_flow = pi_speed_result.y[obs_pipe_idx,:]
pi_flow_obs_flow = pi_flow_result.y[obs_pipe_idx,:]

obs_speed_idx = pumps[0].state_idx
const_obs_speed = no_control_result.y[obs_speed_idx,:]
pd_obs_speed = pd_pressure_result.y[obs_speed_idx,:]
pid_obs_speed = pid_pressure_result.y[obs_speed_idx,:]
pi_speed_obs_speed = pi_speed_result.y[obs_speed_idx,:]
pi_flow_obs_speed = pi_flow_result.y[obs_speed_idx,:]

fire_head = head[3,:]

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{siunitx}\n\\sisetup{per-mode=symbol}\n'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14

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
plt.ylabel(r"Flow (\( \si{\meter^{3} \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
speed_lines = plt.plot(t, speed.T)
#plt.legend(speed_lines, range(len(speed_lines)))
plt.ylabel(r"Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_obs_head.T, 0*"-k", label="Constant")
plt.plot(t, pd_obs_head.T, 0*'--k', label="PD Pressure")
plt.plot(t, pid_obs_head.T, 0*':k', label="PID Pressure")
plt.plot(t, pi_flow_obs_head.T, 0*'-.k', label="PI Flow")
plt.legend()
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_fire_head.T, 0*"-k", label="Constant")
plt.plot(t, pd_fire_head.T, 0*"--k", label="PD Pressure")
plt.plot(t, pid_fire_head.T, 0*":k", label="PID Pressure")
plt.plot(t, pi_flow_fire_head.T, 0*"-.k", label="PI Flow")
plt.legend()
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_obs_flow.T, "-k", label="Constant")
plt.plot(t, pd_obs_flow.T, "--k", label="PD Pressure")
plt.plot(t, pid_obs_flow.T, ":k", label="PID Pressure")
plt.plot(t, pi_flow_obs_flow.T, "-.k", label="PI Flow")
plt.legend()
plt.ylabel(r"Flow (\( \si{\meter^{3} \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, const_obs_speed.T, "-k", label="Constant")
plt.plot(t, pd_obs_speed.T, "--k", label="PD Pressure")
plt.plot(t, pid_obs_speed.T, ":k", label="PID Pressure")
plt.plot(t, pi_flow_obs_speed.T, "-.k", label="PI Flow")
plt.legend()
plt.ylabel(r"Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")








