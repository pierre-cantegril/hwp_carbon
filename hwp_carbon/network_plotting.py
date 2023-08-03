from copy import copy
from typing import Optional
import graphviz as gz
import os

# Test if rendering dot is possible
try:
    dot = gz.Digraph(comment='Test')
    dot.render('test')
    os.remove('test.pdf')
except gz.ExecutableNotFound as e:
    import warnings
    msg = '\n\nTo render Graphviz DOT source code, you also need to install Graphviz'\
          '\nIf you use conda, use:\nconda install python-graphviz' \
          '\nIf you use pip, follow instruction here : https://pypi.org/project/graphviz/' \
          '\n\nYou can still create and simulate carbon flows, but network plotting will not be possible'
    warnings.warn(f"\nWarning:\n{e}{msg}", stacklevel=2)

from .network import CarbonNetwork

def add_pools_to_graph(network: CarbonNetwork, dot: gz.Digraph) -> gz.Digraph:
    """Add pools to a graph"""
    for pool_name, pool in network.pools.items():
        dot.node(f'{pool_name}_{network.pools[pool_name].half_life:.0f}')

    # Force top pools on the same level
    top_pools = network.get_network_top_pools()
    with dot.subgraph(name='Top pools') as sub:
        #sub.graph_attr['rankdir'] = # TB, LR ...https://graphviz.org/docs/attrs/rankdir/
        sub.attr(rank='same')
        for pool_name in top_pools:
            sub.node(f'{pool_name}_{network.pools[pool_name].half_life:.0f}')

    # Force child pool to be on the same rank
    with dot.subgraph(name=f'Top pools - childs') as sub:
        sub.attr(rank='same')
        for pool_name in top_pools:
            pool = network.pools[pool_name]
            for child_name in pool.dst_pools:
                if child_name == 'atmosphere': continue
                sub.node(f'{child_name}_{network.pools[child_name].half_life:.0f}')
    return dot


def add_arcs_to_graph(network: CarbonNetwork, dot: gz.Digraph, kw_args: Optional[dict[str, str]] = None) -> gz.Digraph:
    """Add pools to a graph"""
    if kw_args is None:
        kw_args = {}
    for arc, flow in network.flows.items():
        _kw_args = copy(kw_args)
        if flow.recycling:
            _kw_args['color'] = 'red'
        factor = flow.factor[0] if hasattr(flow.factor, '__getitem__') else flow.factor
        factor = round(factor, 2)
        _arc = tuple([f'{pool_name}_{network.pools[pool_name].half_life:.0f}' for pool_name in arc])
        dot.edge(*_arc)#, taillabel=f'{factor}', labelfontsize='10', **_kw_args)
    return dot


def network_to_dot(network: CarbonNetwork):
    dot = gz.Digraph(comment='Carbon Network', format='png')
    dot = add_pools_to_graph(network, dot)
    dot = add_arcs_to_graph(network, dot)
    return dot

