import numpy as np

import PhysicalElements as pe
import RootFinding as rf
import Simulator as sim
import Controllers
import Disturbances

reservoirs = res = (
    pe.Reservoir(0),
    pe.Reservoir(60),
)

# surface area, conductance, h0
junctions = jncts = (
    pe.FixedOutflowJunction(.001,   0, 74.5),
    pe.FixedOutflowJunction(.001, 200, 64.7),
    pe.FixedOutflowJunction(.001, 300, 60.7),
    pe.FixedOutflowJunction(.001, 200, 69.5),
    pe.FixedOutflowJunction(.001, 300, 64.4),
    pe.FixedOutflowJunction(.001, 300, 60.6),
    pe.FixedOutflowJunction(.001, 240, 60),
)

fire_start_time = 100
fire_valve_opening_duration = 10
fire_disturbance = Disturbances.OutflowValveOpeningDisturbance(junctions[3], fire_start_time, fire_valve_opening_duration)
junctions[3].disturbance = fire_disturbance.disturbance_builder()

# (start, end), length (m), diameter (m), darcy-weisback friction factor, local loss coefficient, q0
pipes = (
    pe.Pipe((res[0], jncts[0]),     10, .30,     0,  0, 0.186),
    pe.Pipe((jncts[0], jncts[1]), 1000, .25, 0.015, .9, .0875),
    pe.Pipe((jncts[1], jncts[2]), 1000, .20, 0.015, .9, .0318),
    pe.Pipe((jncts[0], jncts[3]), 1000, .30, 0.015, .9, .0989),
    pe.Pipe((jncts[1], jncts[4]), 1000, .25, 0.015, .9, .0155),
    pe.Pipe((jncts[2], jncts[5]), 1000, .20, 0.015, .9, .0058),
    pe.Pipe((jncts[3], jncts[4]), 1000, .30, 0.015, .9, .0989),
    pe.Pipe((jncts[4], jncts[5]), 1000, .25, 0.015, .9, .0542),
    pe.Pipe((jncts[4], jncts[6]), 1000, .20, 0.015, .9, .0334),
    pe.Pipe((jncts[5], res[1]),   1000, .30, 0.015, .9, .0340),
    pe.Pipe((jncts[6], res[1]),   1000, .25, 0.015, .9, .00117),
)

# link id, a, b, inertia, Cq, diameter (m), damping, w0 (rad/s)
pumps = (
    pe.Pump(pipes[0], -120 / 300 * 5 / 8, 120 / 300**2, 2e-1, 1/300, .5, 0.1, 254.57),
)

input_guess = [330.167]

initial_condition_finder = icf = rf.WDN_RootFinder(reservoirs, junctions, pipes, pumps, input_guess)
xx = [s.x0 for s in icf.states]

# restrict flows to be positive
def validator(state):
    head, flow, speed = np.split(state, np.cumsum([len(jncts), len(pipes)]))

    if (head < 20).any() or (flow < 0).any() or (speed < 0).any():
        return False

    return True

bounds = tuple((60, None) for jnct in junctions) + tuple((0.2, None) for pipe in pipes) + \
    tuple((0, None) for pump in pumps) + tuple((0, None) for pump in pumps)
    
solver_weights = np.array(tuple(1 for jnct in junctions) + tuple(100 for pipe in pipes) + \
    tuple(1 for pump in pumps) + tuple(1 for pump in pumps))

#initial_condition_result = icr = initial_condition_finder.find_equilibrium_constrained(weights=solver_weights, bounds=bounds)
initial_condition_result = icr = initial_condition_finder.find_equilibrium(equilibrium_validator=None, verbosity=1)
ic = initial_condition_result.x

controller = Controllers.ConstantController(xx[-1])
simulator = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [controller])

sim_result = simulator.simulate([0, 300], xx[:-1])


### Plotting

t = sim_result.t
head, flow, speed, *aug = np.split(sim_result.y, np.cumsum([len(junctions), len(pipes), len(pumps)]))

fire_head = head[3,:]

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{siunitx}'

plt.figure()
plt.plot(t, head.T)
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, flow.T)
plt.ylabel(r"Flow (\( \si{\meter^{3} /per /second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(t, speed.T)
plt.ylabel(r"Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

# x0_not_eq = False
# shutoff_pumps = False
# parameter_error = True and False # for now
# T0 = 130
# do_plotting = True

# def plotting(sim_result, x_eq, u_eq, disturbances, plt):
#     t, x = sim_result.t, sim_result.y
#     x = x[:,t>100]
#     t = t[t>100]
#     t0 = t[0]
#     t = t - t0
#     h, q, w, *_ = np.split(x, np.cumsum([len(type2_nodes), len(links), len(pumps)]))
    
#     h_fire = h[3,:]
#     h_feedback = h[0,:]
#     q_pump = q[0,:]
#     fire_demand = [-disturbances[3](time + t0, state) for time, state in zip(t, x.T)]
#     pump_speed = w.T
    
#     plt.figure()
#     plt.plot(t, h_fire.T)
#     plt.ylabel(r"Head (\( \si{\meter} \))")
#     plt.xlabel(r"Time (\( \si{\second} \))")
#     #plt.legend()
    
#     plt.figure()
#     plt.plot(t, h_feedback.T, label="Feedback Node")
#     plt.ylabel(r"Head (\( \si{\meter} \))")
#     plt.xlabel(r"Time (\( \si{\second} \))")
#     #plt.legend()
    
#     plt.figure()
#     plt.plot(t, q_pump.T)
#     plt.ylabel(r"Flow (\( \si{ \meter^{3} \per \second} \))")
#     plt.xlabel(r"Time (\( \si{\second} \))")
#     #plt.legend()
    
#     plt.figure()
#     plt.plot(t, fire_demand)
#     plt.ylabel(r"Demand Flow (\( \si{ \meter^{3} \per \second} \))")
#     plt.xlabel(r"Time (\( \si{\second} \))")
#     #plt.legend()
    
#     plt.figure()
#     plt.plot(t, pump_speed)
#     plt.ylabel(r"Radial Speed (\( \si{ \radian \per \second} \))")
#     plt.xlabel(r"Time (\( \si{\second} \))")
#     #plt.legend()
