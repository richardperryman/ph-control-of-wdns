import numpy as np

import PhysicalElements

def state_index_generator(start=0):
    idx = start
    while True:
        yield idx
        idx += 1

def get_head(wdn_node, x):
    if isinstance(wdn_node, PhysicalElements.Reservoir):
        return wdn_node.head
    return x[wdn_node.state_idx]

def rough_convert_HW_to_DW(C):
    return 10.67 * 9.81 * np.pi**2 / 8 / C ** 1.852

def print_states(sim_result, t_idx, n_nodes, n_pipes, n_pumps):
    head, flow, speed, *aug = np.split(sim_result.y, np.cumsum([n_nodes, n_pipes, n_pumps]))
    print(f"Heads: {head[:,t_idx]}")
    print(f"Flows: {flow[:,t_idx]}")
    print(f"Speeds: {speed[:,t_idx]}")