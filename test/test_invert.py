'''
Data inversion tests.
'''

import os
import unittest
import json

import numpy as np

import pysotope.ratios as ratios
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
        '''Calculate reference exponent in non log space'''
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


class TestInvertData(unittest.TestCase):
    '''
    Test inversion. This is not a real unit-test since invert_data is a
    complicated function and should be tested after all of its constituents.
    '''
    def setUp(self):
        f_spike = 0.35
        self.pn = 0.1
        self.nn = -0.1
        self.pi = 0.1
        self.ni = -0.1
        standard_ratios = ratios.calc_spec_ratios(
            'standard',
            'report_fracs',
            SPEC)

        def get_fract_mix(alpha, beta):
            fract_std = ratios.fractionate_abund(
                SPEC['standard'], 'report_fracs',
                alpha, SPEC)
            mix = ratios.mix_abund(
                SPEC['spike'],
                fract_std,
                f_spike)
            fract_mix = ratios.fractionate_abund(
                mix, 'report_fracs',
                beta, SPEC)
            return fract_mix
        
        self.mix = ratios.mix_abund(
            SPEC['spike'],
            SPEC['standard'],
            f_spike)
        self.mix = ratios.mix_abund(
            SPEC['spike'],
            SPEC['standard'],
            f_spike)

        # 0n__ zero nat_frac
        # pn__ positive nat frac
        # nn__ negative nat frac
        # __pi positive internal frac
        # __ni negative internal frac
        self.mix_0n0i = get_fract_mix(      0,       0)
        self.mix_0npi = get_fract_mix(      0, self.pi)
        self.mix_0nni = get_fract_mix(      0, self.ni)
        self.mix_pn0i = get_fract_mix(self.pn,       0)
        self.mix_pnpi = get_fract_mix(self.pn, self.pi)
        self.mix_pnni = get_fract_mix(self.pn, self.ni)
        self.mix_nn0i = get_fract_mix(self.nn,       0)
        self.mix_nnpi = get_fract_mix(self.nn, self.pi)
        self.mix_nnni = get_fract_mix(self.nn, self.ni)

    def test_mix(self):
        '''Test mixture without any interference'''
        row = ratios.row_from_abund(self.mix, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],        0, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0],       0, delta=1e-8)

    def test_0n0i(self):
        row = ratios.row_from_abund(self.mix_0n0i, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],        0, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0],       0, delta=1e-8)

    def test_0npi(self):
        row = ratios.row_from_abund(self.mix_0npi, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],  self.pi, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0],       0, delta=1e-8)

    def test_0nni(self):
        row = ratios.row_from_abund(self.mix_0nni, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],  self.ni, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0],       0, delta=1e-8)

    def test_pn0i(self):
        row = ratios.row_from_abund(self.mix_pn0i, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],        0, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0], self.pn, delta=1e-8)

    def test_pnpi(self):
        row = ratios.row_from_abund(self.mix_pnpi, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],  self.pi, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0], self.pn, delta=1e-8)

    def test_pnni(self):
        row = ratios.row_from_abund(self.mix_pnni, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],  self.ni, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0], self.pn, delta=1e-8)

    def test_nn0i(self):
        row = ratios.row_from_abund(self.mix_nn0i, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],        0, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0], self.nn, delta=1e-8)

    def test_nnpi(self):
        row = ratios.row_from_abund(self.mix_nnpi, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],  self.pi, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0], self.nn, delta=1e-8)

    def test_nnni(self):
        row = ratios.row_from_abund(self.mix_nnni, SPEC)
        results = invert_data([row], SPEC)
        self.assertAlmostEqual(results['beta_ins'][0],  self.ni, delta=1e-8)
        self.assertAlmostEqual(results['alpha_nat'][0], self.nn, delta=1e-8)


if __name__ == '__main__':
    unittest.main()
