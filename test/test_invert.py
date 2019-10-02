'''
Data inversion tests.
'''

import os
import unittest
import json

import numpy as np

from pysotope.invert import (
        calc_stat,
        create_interf_corr,
        create_interf_corrfuncs,
        create_row_lookup,
        exp_corr,
        gen_filter_function,
        generate_raw_labels,
        get_interf_corr,
        get_interf_corr_fun,
        get_reduction_fun,
        invert_data,
        summarise_data,
        )

TEST_DIR = os.path.split(__file__)[0]
with open(os.path.join(TEST_DIR, 'Cr-test-scheme.json'), 'r') as file_handle:

    SPEC = json.load(file_handle)


class TestExpCorr(unittest.TestCase):
    '''
    Tests for exponential fractionation correction
    '''
    def setUp(self):
        self.init_ratio = 1.0
        self.mass_ratio = 0.1

    def ref_fun(self, frac_fact):
        exponent = - frac_fact * np.log(self.mass_ratio)
        result = self.init_ratio * np.exp(exponent)
        return result

    def test_unfract(self):
        '''Test alpha = 0'''
        frac_fact = 0
        result = exp_corr(self.init_ratio, self.mass_ratio, frac_fact)
        self.assertAlmostEqual(self.init_ratio, result)

    def test_posfract(self):
        '''Test alpha > 0'''
        frac_fact = 0.1
        result = exp_corr(self.init_ratio, self.mass_ratio, frac_fact)
        ref = self.ref_fun(frac_fact)
        self.assertAlmostEqual(ref, result)

    def test_negfract(self):
        '''Test alpha > 0'''
        frac_fact = -0.1
        result = exp_corr(self.init_ratio, self.mass_ratio, frac_fact)
        ref = self.ref_fun(frac_fact)
        self.assertAlmostEqual(ref, result)


if __name__ == '__main__':
    unittest.main()
