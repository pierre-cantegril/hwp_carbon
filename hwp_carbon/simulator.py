import copy
from typing import Sequence

import numpy as np

from data_classes import PoolName
from exceptions import NetworkLoopError
from network import CarbonNetwork
from utils import adjust_sequence_length


def run_simulation(network : CarbonNetwork, carbon_inputs: dict[PoolName, Sequence], steps: int = None,
                   flow_lo_threshold: float = 0.001, verbose: bool = False, _first_run: bool = True) -> None:
    """Run the simulation of carbon decomposition in the network"""
    # Handle args
    if not steps:
        steps = max([len(pool_inputs) for _, pool_inputs in carbon_inputs.items()])

    for pool_name, pool_input in carbon_inputs.items():
        if pool_name not in network.pools:
            raise ValueError(f'the pool {pool_name} from pools_inputs arg is not in the network pools')

        # Change sequence to numpy.array
        carbon_inputs[pool_name] = np.array(pool_input)

        # Check length of initial_carbon_stock. expand or slice if the length != steps
        carbon_inputs[pool_name] = adjust_sequence_length(carbon_inputs[pool_name], steps)

    if verbose: print(f'{carbon_inputs=}')
    if verbose: print(f'{steps=}')

    for pool_name in network.pools:
        if pool_name not in carbon_inputs:
            carbon_inputs[pool_name] = np.zeros(steps)

    # Prepare network if not in recursion loop
    if _first_run:
        network.user_carbon_inputs = copy.deepcopy(carbon_inputs)
        is_looping, msg_looping = network.search_network_loops()
        if is_looping:
            raise NetworkLoopError(msg_looping)
        network.set_decay_func_as_radioactive()
        network.set_flows_for_simulation(steps)
        network.set_pools_for_simulation(steps)

    # Simulate
    for pool_name in network.sim_sequence:
        if verbose: print(f'\n{pool_name=}', end=' : ')
        if carbon_inputs[pool_name].sum() == 0:
            if verbose: print('no carbon input')
            continue
        pool = network.pools[pool_name]
        carbon_input = np.append(carbon_inputs[pool_name], 0)

        if pool.half_life == 0:  # Instant decomposition
            if verbose: print('instant decomposition')
            pool_carbone_after_decay = np.zeros(carbon_input.shape)
            pool.carbon_stock += pool_carbone_after_decay[1:]
            decayed_carbon = carbon_inputs[pool_name]
        elif np.isnan(pool.half_life) or pool.half_life < 0:  # No decomposition
            if verbose: print('no decomposition')
            pool_carbone_after_decay = carbon_input.cumsum()
            pool.carbon_stock += pool_carbone_after_decay[:-1]
            decayed_carbon = np.zeros(steps)
        else:  # Standard decomposition
            if carbon_input.max() < flow_lo_threshold:
                print('lower threshold for standard decomposition is not reached. No decomposition.')
                continue
            if verbose: print('standard decomposition')
            pool_carbone_after_decay = (pool.decay_func(carbon_input) - carbon_input)
            pool.carbon_stock += pool_carbone_after_decay[1:]
            decayed_carbon = ((pool_carbone_after_decay + carbon_input) - np.append(pool.carbon_stock, 0))[:-1]
        if verbose: print(f'{carbon_input=}')
        if verbose: print(f'{pool_carbone_after_decay=}')
        if verbose: print(f'{pool.carbon_stock=}')
        if verbose: print(f'{decayed_carbon=}')

        for child_name in pool.dst_pools:
            arc = (pool_name, child_name)
            flow = network.flows[arc]
            flow_from_pool_to_child = (decayed_carbon * flow.factor)
            if flow.delay:
                flow_from_pool_to_child = np.insert(flow_from_pool_to_child,
                                                    0, [0] * flow.delay)[:-flow.delay]
            flow.values += flow_from_pool_to_child
            if flow.recycling:

                if verbose: print(f'Recursion loop started : {arc}')
                # network.run_simulation(
                #     _carbon_inputs={child_name: flow_from_pool_to_child},
                #     steps=steps,
                #     flow_lo_threshold=flow_lo_threshold, verbose=verbose, _first_run=False)
                if verbose: print(f'Recursion loop ended : {arc}')
            else:
                carbon_inputs[child_name] += flow_from_pool_to_child

            if verbose: print(f'  {child_name=}')
            if verbose: print(f'    {flow_from_pool_to_child=}')
            if verbose: print(f'    {carbon_inputs[child_name]=}')

    if verbose: print('\nSummary of pool stocks:')
    if verbose: print('\n'.join([f'{pool_name=} : {pool.carbon_stock}' for pool_name, pool in network.pools.items()]))
    return