import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from hwp_carbon.hwp_carbon.network import CarbonNetwork
from hwp_carbon.hwp_carbon.utils import excel_to_carbonnetwork_init_data

# Set basic data for unittesting
test_data = './test_data.xlsx'
init_data = excel_to_carbonnetwork_init_data(test_data)
net = CarbonNetwork(init_data)
inputs = {'paper': [100, 0, 0, 0, 0, 0, 0], 'veneer': [0]}


class TestInputs(unittest.TestCase):
    def test_inputs_alteration(self):
        """Test if user inputs is modified"""
        _inputs = deepcopy(inputs)
        net.run_simulation(inputs)
        self.assertEqual(inputs, _inputs)
        pass


class TestNetwork(unittest.TestCase):
    def test_flows_attr_type(self):
        """Test if the type of Flow objects are the good ones"""
        for arc, flow in net.flows.items():
            # self.assertIsInstance(arc, tuple)
            self.assertIsInstance(flow.factor, np.ndarray)
            self.assertIsInstance(flow.values, np.ndarray)
            self.assertIsInstance(flow.delay, int)
            self.assertIsInstance(flow.recycling, bool)



class TestNetworkOutputs(unittest.TestCase):
    def test_network_integrity(self):
        """Test if there is unwanted carbon leaks or infiltrations in the network"""
        net.run_simulation(inputs)

        sum_carbon = np.zeros(net.sim_steps)

        # Get carbon stocked in pools
        sum_carbon += pd.DataFrame(net.get_pools_attr('carbon_stock')).T.sum()

        # Get carbon stocked in delayed flows
        df_flow = net.get_flows_attr('values', as_dataframe=True)
        delayed_flows = [arc for arc, flow in net.flows.items() if flow.delay]
        carbon_in_flow = df_flow[df_flow.index.isin(delayed_flows)].sum()
        sum_carbon += carbon_in_flow

        cumsum_inputs = sum(net.user_carbon_inputs.values()).cumsum()

        np.testing.assert_almost_equal(cumsum_inputs, sum_carbon, decimal=6)


if __name__ == '__main__':
    unittest.main()
