import copy
from dataclasses import dataclass, field, InitVar
from typing import Any, Callable, Sequence, Optional

import numpy as np
from collections import defaultdict

import pandas as pd

from .data_classes import Pool, Flow, PoolName
from .utils import cyclic, adjust_array
from .exceptions import NetworkLoopError
from .decay import radioactive_decay_func_numpy


@dataclass
class CarbonNetwork:
    init_data: InitVar[dict[str, dict]]
    pools: dict[PoolName, Pool] = field(init=False)
    arcs: tuple = field(init=False)
    flows: dict[tuple[PoolName, PoolName], Flow] = field(init=False)
    user_carbon_inputs: Optional[Sequence] = field(init=False)
    sim_sequence: Optional[Sequence] = field(init=False)
    sim_steps: Optional[int] = field(init=False)

    def __post_init__(self, init_data: dict[str, dict]):
        self.pools = {pool_name: Pool(**pool_data) for pool_name, pool_data in init_data['pools'].items()}
        self.flows = {arc: Flow(**arc_data) for arc, arc_data in init_data['arcs'].items()}
        self.arcs = tuple(self.flows.keys())

        # Compile src and dst pools for each pool
        for src_pool, dst_pool in self.arcs:
            self.pools[src_pool].dst_pools.add(dst_pool)
            self.pools[dst_pool].src_pools.add(src_pool)

        is_cyclic, msg = self.is_cyclic()
        if is_cyclic:
            raise NetworkLoopError(msg)
        self.sim_sequence = self.get_pool_simulation_sequence()

    def get_pools_attr(self, attr: str, as_dataframe: bool = False) -> dict[PoolName, Any] | pd.DataFrame:
        """Return a specific attr of pool class for each pool"""
        try:
            pools_attr = {pool_name: pool.__dict__[attr] for pool_name, pool in self.pools.items()}
        except KeyError:
            raise AttributeError(f"'Pool' object has no attribute '{attr}'")
        if as_dataframe:
            pools_attr = pd.DataFrame(pools_attr).T
            pools_attr.index.names = ['pool_name']
        return pools_attr

    def get_flows_attr(self, attr: str, as_dataframe: bool = False) -> dict[tuple[str, str], Any] | pd.DataFrame:
        """Return a specific attr of flow class for each flow"""
        try:
            flow_attr = {arc: flow.__dict__[attr] for arc, flow in self.flows.items()}
        except KeyError:
            raise AttributeError(f"'Flow' object has no attribute '{attr}'")
        if as_dataframe:
            flow_attr = pd.DataFrame(flow_attr).T
            flow_attr.index.names = ('src_pool', 'dst_pool')
        return flow_attr

    def get_network_top_pools(self, verbose: bool = False):
        """Returns all pools without src_pools or with only with recycling parents"""
        top_pools = [pool_name for pool_name, pool in self.pools.items()
                     if not pool.src_pools
                     or all([self.flows[(parent_name, pool_name)].recycling is True
                             for parent_name in pool.src_pools])]
        return top_pools

    def get_network_bottom_pools(self, verbose: bool = False):
        """Returns all pools without dst_pools or with only with recycling childs"""
        bot_pools = [pool_name for pool_name, pool in self.pools.items()
                     if not pool.dst_pools
                     or all([self.flows[(pool_name, child_name)].recycling is True
                             for child_name in pool.dst_pools])]
        return bot_pools

    def is_cyclic(self) -> tuple[bool, str]:
        """Search for loops in the whole network. Return True is graph is cyclic + some readable message for the user
        Recycling flows are excluded from this testing"""
        graph_without_recycling = {pool_name: [child_name
                                               for child_name in pool.dst_pools
                                               if not self.flows[pool_name, child_name].recycling]
                                   for pool_name, pool in self.pools.items()}
        return cyclic(graph_without_recycling)

    def get_pool_simulation_sequence(self, verbose: bool = False) -> list[PoolName]:
        """Get an ordered list of pools for simulation.
        Ensure that a pool has no child pool before it in the list unless it's a recycling flow"""

        sim_sequence = list(self.get_network_top_pools())
        all_child_added_to_sequence = []

        while len(sim_sequence) < len(self.pools):
            for pool_name in sim_sequence:
                if len(sim_sequence) == len(self.pools):
                    return sim_sequence
                if pool_name in all_child_added_to_sequence: continue
                if verbose: print(f'Searching from {pool_name}')
                pool = self.pools[pool_name]
                fully_searched = True
                for child_name in pool.dst_pools:
                    if child_name in sim_sequence:
                        if verbose: print(f'  {child_name} already in sim_sequence')
                        continue
                    child = self.pools[child_name]
                    if all([parent_name in sim_sequence for parent_name in child.src_pools]):
                        if verbose: print(f'  Appending {child_name} to sim_sequence')
                        sim_sequence.append(child_name)
                    else:
                        if verbose: print(f'  All parents pool of {child_name} are not in sim_sequence')
                        fully_searched = False
                all_child_added_to_sequence.append(pool_name)
                if fully_searched:
                    all_child_added_to_sequence.append(pool_name)
        return sim_sequence

    def set_decay_func_as_radioactive(self, _radioactive_decay_func: Callable = radioactive_decay_func_numpy) -> None:
        """Set a standard radioactive decay function for each pool"""
        for pool_name, pool in self.pools.items():
            pool.decay_func = _radioactive_decay_func(pool.half_life)

    def set_flows_for_simulation(self, steps: int) -> None:
        """Reset the flow values and ensure that values and factor attributes length egal steps"""
        for arc, flow in self.flows.items():
            flow.values = np.zeros(steps)
            flow.factor = adjust_array(flow.factor, steps, interpolate_nan=True)

    def set_pools_for_simulation(self, steps: int) -> None:
        """Reset the carbon_stock attribute for each pool"""
        for pool_name, pool in self.pools.items():
            pool.carbon_stock = np.zeros(steps)
            pool.substitution_factor = adjust_array(pool.substitution_factor, steps, interpolate_nan=True)
            pool.substitution = np.zeros(steps)

    def run_simulation(self, carbon_inputs: dict[PoolName, np.ndarray | Sequence], steps: int = None,
                       flow_lo_threshold: float = 0.001, verbose: bool = False, _first_run: bool = True) -> None:
        """Run the simulation of carbon decomposition in the network"""
        # Handle args
        _carbon_inputs = copy.deepcopy(carbon_inputs) # To not alter user inputs
        if not steps:
            steps = max([len(pool_inputs) for _, pool_inputs in _carbon_inputs.items()])
        self.sim_steps = steps

        for pool_name, pool_input in _carbon_inputs.items():
            if pool_name not in self.pools:
                raise ValueError(f'the pool {pool_name} from carbon_inputs arg is not in the network pools')

            # Change sequence to numpy.array
            _carbon_inputs[pool_name] = np.array(pool_input)

            # Check length of initial_carbon_stock. expand or slice if the length != steps
            _carbon_inputs[pool_name] = adjust_array(_carbon_inputs[pool_name], steps,
                                                     interpolate_nan=False, value_to_add=0)

        if verbose and _first_run: print(f'{_carbon_inputs=}')
        if verbose and _first_run: print(f'{steps=}')

        for pool_name in self.pools:
            if pool_name not in _carbon_inputs:
                _carbon_inputs[pool_name] = np.zeros(steps)

        # Prepare network if not in recursion loop
        if _first_run:
            self.user_carbon_inputs = copy.deepcopy(_carbon_inputs)
            self.set_decay_func_as_radioactive()
            self.set_flows_for_simulation(steps)
            self.set_pools_for_simulation(steps)

        # Simulate
        for pool_name in self.sim_sequence:
            if _carbon_inputs[pool_name].sum() == 0:
                continue
            if verbose: print(f'\n{pool_name=}', end=' : ')
            pool = self.pools[pool_name]
            pool.substitution += _carbon_inputs[pool_name] * pool.substitution_factor
            carbon_input = np.append(_carbon_inputs[pool_name], 0)

            if pool.half_life == 0:  # Instant decomposition
                if verbose: print('instant decomposition')
                pool_carbone_after_decay = np.zeros(carbon_input.shape)
                pool.carbon_stock += pool_carbone_after_decay[1:]
                decayed_carbon = _carbon_inputs[pool_name]
            elif np.isnan(pool.half_life) or pool.half_life < 0:  # No decomposition
                if verbose: print('no decomposition')
                pool_carbone_after_decay = carbon_input.cumsum()
                pool.carbon_stock += pool_carbone_after_decay[:-1]
                decayed_carbon = np.zeros(steps)
            else: # Standard decomposition
                if carbon_input.max() < flow_lo_threshold:
                    print('lower threshold for standard decomposition is not reached. No decomposition.')
                    continue
                if verbose: print('standard decomposition')
                pool_carbone_after_decay = (pool.decay_func(carbon_input).astype('float64') - carbon_input)
                if verbose: print(f'{pool_carbone_after_decay=}')
                pool.carbon_stock += pool_carbone_after_decay[1:]
                decayed_carbon = ((pool_carbone_after_decay + carbon_input) - np.append(pool_carbone_after_decay[1:], 0))[:-1]

            if verbose: print(f'{_carbon_inputs[pool_name]=}')
            if verbose: print(f'{pool.carbon_stock=}')
            if verbose: print(f'{decayed_carbon=}')

            for child_name in pool.dst_pools:
                arc = (pool_name, child_name)
                flow = self.flows[arc]
                flow_from_pool = (decayed_carbon * flow.factor)
                flow_to_child = flow_from_pool if not flow.delay else np.insert(
                    flow_from_pool, 0, [0] * flow.delay)[:-flow.delay]

                flow.values += flow_from_pool
                if flow.recycling:

                    if verbose: print(f'🔁 Recursion loop started : {arc}')
                    self.run_simulation(
                        carbon_inputs={child_name: flow_to_child},
                        steps=steps,
                        flow_lo_threshold=flow_lo_threshold, verbose=verbose, _first_run=False)
                    if verbose: print(f'🔁 Recursion loop ended : {arc}')
                else:
                    _carbon_inputs[child_name] += flow_to_child

                if verbose: print(f'  {child_name=}')
                if verbose: print(f'    {flow_from_pool=}')
                if verbose: print(f'    {flow_to_child=}')
                if verbose: print(f'    {_carbon_inputs[child_name]=}')

        if verbose and _first_run: print('\nSummary of pool stocks:')
        if verbose and _first_run: print('\n'.join([f'{pool_name=} : {pool.carbon_stock.round(2)}' for pool_name, pool in self.pools.items()]))
        return

#
# if __name__ == '__main__':
#     from utils import read_excel
#     pools, arcs_data = read_excel('./test_data.xlsx')
#     net = CarbonNetwork(pools=pools, arcs_data=arcs_data)
#     print(f'{net.is_cyclic()=}')
#     print(net.pools)