# Adapted from J. D. Nault's PhD thesis: http://hdl.handle.net/1807/80861
# Taken from Section 5.6.1, figure 5.3, page 88

import numpy as np

import PhysicalElements as pe
import RootFinding as rf
import Simulator as sim
import Controllers
import Disturbances

from HelperFunctions import rough_convert_HW_to_DW as HW_to_DW

valve_close_begin = 100
valve_close_time = 100
shutoff_pump_time = valve_close_begin + valve_close_time - 1

reservoirs = res = (
    pe.Reservoir(100),
    pe.Reservoir(150)
)

junctions = jncts = (
    pe.Junction(np.pi * .5 ** 2, 0, 167),
)

pipes = (
    pe.Pipe((res[0], jncts[0]), 50, .5, HW_to_DW(120), .0, .62),
    pe.Pipe((jncts[0], res[1]), 1000, .5, HW_to_DW(120), .0, .62),
)
pipes = (
    pe.HW_Pipe((res[0], jncts[0]), 50, .5, (120), .0, .62),
    pe.HW_Pipe((jncts[0], res[1]), 1000, .5, (120), .0, .62),
)

pipes[0].disturbance = Disturbances.ShutoffValveDisturbance(pipes[0], 10, valve_close_begin, valve_close_time).disturbance_builder()

pumps = (
    pe.Pump(pipes[0], -120 / 300 * 5 / 8, 120 / 300**2, 2e-1, 1/300, .5, 0, 300),
)

input_guess = [300]

initial_condition_finder = icf = rf.WDN_RootFinder(reservoirs, junctions, pipes, pumps, input_guess)

initial_condition_result = icr = initial_condition_finder.find_equilibrium()
ic = initial_condition_result.x

controller = Controllers.ConstantController(ic[-1])
closing_time_range = (0, shutoff_pump_time)
closing_sim = sim.WDN_Simulator(reservoirs, junctions, pipes, pumps, [controller])

closing_result = closing_sim.simulate(closing_time_range, ic[:-1])

remainder_sim = sim.WDN_Simulator(reservoirs[1:], junctions, pipes[1:], [], [])

remainder_time_range = (closing_result.t[-1], 1000)
remainder_result = remainder_sim.simulate(remainder_time_range, closing_result.y[[0, 2],-1]) #TODO try max step less cursed?


### Plotting

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}\n\\usepackage{siunitx}'

t = np.concatenate([closing_result.t, remainder_result.t])
tank_head_c, pump_flow_c, pipe_flow_c, pump_speed_c = closing_result.y
tank_head_r, pipe_flow_r = remainder_result.y

tank_head = np.concatenate([tank_head_c, tank_head_r])
pipe_flow = np.concatenate([pipe_flow_c, pipe_flow_r])

plt.figure()
plt.plot(t, tank_head)
plt.ylabel(r"Head (\( \si{\meter} \))")
plt.xlabel(r"Time (\( \si{\second} \))")

plt.figure()
plt.plot(closing_result.t, pump_flow_c, label="Pump flow")
plt.plot(t, pipe_flow, label="Outflow")
plt.ylabel(r"Flow (\( \si{ \meter^{3} \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")
plt.legend()

plt.figure()
plt.plot(closing_result.t, pump_speed_c)
plt.ylabel(r"Radial Speed (\( \si{\radian \per \second} \))")
plt.xlabel(r"Time (\( \si{\second} \))")