import unittest
import numpy as np
import consumerUtility as cu

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class MyTestCase(unittest.TestCase):
    def test1(self, num_goods=20, num_trials=1000, tolerance=1e-2):
        """
        Test num_trials many i.i.d. random valuations and then check if properties are satisfied
            num_goods: the number of goods. Default is 20.
            num_trials: the number of random, i.i.d. markets.
        """
        for trial in range(num_trials):            
            # Vector of valuations: |goods|
            valuations = np.random.rand(num_goods)
            prices = np.random.rand(num_goods) + 1
            # Budgets of buyers: |buyers|
            budget = np.random.rand(1) + 1
            
            # Utility level of buyers
            util_level = np.random.rand(1) + 2

            # Elasticity of substitution for CES
            rho = 0.5

            # Linear utilty function
            util_linear = lambda x: x.T @ valuations

            indirect_util_linear, marshallian_linear = cu.get_custom_ump(prices, budget, util_linear)
            expend_linear, hicksian_linear = cu.get_custom_emp(prices, util_level, util_linear)

            assert np.abs(util_linear(marshallian_linear) - indirect_util_linear) < tolerance
            assert (np.abs(cu.get_custom_hicksian_demand(prices, indirect_util_linear, util_linear) - marshallian_linear) < tolerance).all()
            assert (np.abs(cu.get_custom_marshallian_demand(prices, expend_linear, util_linear) - hicksian_linear) < tolerance).all()
            
            # CES utility function


            indirect_util_ces, marshallian_ces = cu.get_ces_ump(prices, budget, valuations, rho)
            expend_ces, hicksian_ces = cu.get_ces_emp(prices, util_level, valuations, rho)

            assert np.abs(cu.get_ces_utility(marshallian_ces, valuations, rho) - indirect_util_ces) < tolerance
            assert (np.abs(cu.get_ces_hicksian_demand(prices, indirect_util_ces, valuations, rho) - marshallian_ces) < tolerance).all()
            assert (np.abs(cu.get_ces_marshallian_demand(prices, expend_ces, valuations, rho) - hicksian_ces) < tolerance).all()            
            
            # Leontief Utility 

            indirect_util_leontief, marshallian_leontief = cu.get_leontief_ump(prices, budget, valuations)
            expend_leontief, hicksian_leontief = cu.get_leontief_emp(prices, util_level, valuations)

            assert np.abs(cu.get_leontief_utility(marshallian_leontief, valuations) - indirect_util_leontief) < tolerance
            assert (np.abs(cu.get_leontief_hicksian_demand(prices, indirect_util_leontief, valuations) - marshallian_leontief) < tolerance).all()
            assert (np.abs(cu.get_leontief_marshallian_demand(prices, expend_leontief, valuations) - hicksian_leontief) < tolerance).all()            

            # Cobb-Douglas Utility 

            indirect_util_CD, marshallian_CD = cu.get_CD_ump(prices, budget, valuations)
            expend_CD, hicksian_CD = cu.get_CD_emp(prices, util_level, valuations)

            assert np.abs(cu.get_CD_utility(marshallian_CD, valuations) - indirect_util_CD) < tolerance
            assert (np.abs(cu.get_CD_hicksian_demand(prices, indirect_util_CD, valuations) - marshallian_CD) < tolerance).all()
            assert (np.abs(cu.get_CD_marshallian_demand(prices, expend_CD, valuations) - hicksian_CD) < tolerance).all()            


if __name__ == '__main__':
    unittest.main()
