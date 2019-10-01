'''
Double spike inversion routines for pysotope.
'''

# pysotope - a package for inverting double spike isotope analysis data
#
#     Copyright (C) 2018 Trygvi Bech Arting
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


# encoding: utf-8


from math import ceil
import warnings
from collections import OrderedDict

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats

from pysotope.typedefs import (
        List,
        Dict,
        Tuple,
        Any,
        Callable,
        Spec,
        RowMx,
        Row,
        IntensRow,
        RatioRow,
        Column,
        AlphaBetaLambda,
        )


np.seterr(all='raise')


def exp_corr(
        R_initial: float,
        R_mass: float,
        frac_fact: float,
        ) -> float:
    '''
    Function for exponential mass bias correction.
    '''
    with np.errstate(invalid='raise'):
        try:
            result = np.exp(np.log(R_initial) - frac_fact * np.log(R_mass))
        except FloatingPointError:
            result = np.array(np.nan)

    # return R_initial * np.exp(-frac_fact * np.log(R_mass))
    return result


def gen_interf_func(
        m_corr: str,
        m_ref: str,
        interf_elem: str,
        file_spec: Spec,
        ) -> Callable[[float, float], float]:
    '''
    Generates one interference correction function based based on info
    from the spec_file "masses"; "nat_ratios"; and m_corr, m_ref and 
    interf_elem from "used_isotopes". Returns a function which
    takes the reference isotope raw_signal and instrumental fractionation
    factor producing the calculated interfering signal.
    '''
    mass_ref = file_spec['masses'][m_ref+interf_elem]
    mass_interf = file_spec['masses'][m_corr+interf_elem]
    R_mass = mass_ref/mass_interf
    R_abund_label = '{1}{2}/{0}{2}'.format(m_corr, m_ref, interf_elem)
    R_abund = file_spec['nat_ratios'][R_abund_label]

    def corr_intensity(ref_raw, beta_ins):
        return ref_raw / exp_corr(R_abund, R_mass, beta_ins)

    return corr_intensity


def create_row_lookup(
        index: int,
        ) -> Callable[[IntensRow, Any], float]:
    '''
    Function generator for element lookup. Takes an index and returns 
    a function which takes a row of raw analyses and returns the value
    at value index.
    '''
    def raw_lookup(
                row: IntensRow,
                beta: Any = None
                ) -> float:
        '''
        Takes a row, and returns value at index: {}
        '''.format(index)
        return row[index]
    return raw_lookup


def create_interf_corr(
        m_to_corr: int,
        file_spec: Spec
        ) -> Callable[[IntensRow, float], float]:
    '''
    Function generator for a single mass interference correction.
    Returns a function which takes a row and mass bias factor, 
    and returns a corrected value based on all interferences listed in
    "used_isotopes".
    '''
    interf = file_spec['used_isotopes'][m_to_corr]
    interf_funcs = dict()
    for m_ref, elem in interf:
        interf_funcs[m_ref] = gen_interf_func(
            m_to_corr,
            m_ref,
            elem,
            file_spec)

    def interf_corr_row(
                row: IntensRow,
                beta: float,
                ) -> float:
        '''
        Corrects {} for interferences listed in "used_isotopes".
        '''.format(m_to_corr)
        i_m_to_corr = file_spec['cycle_columns'][m_to_corr]
        value = row[i_m_to_corr]
        for k, fun in interf_funcs.items():
            i_m_reference = file_spec['cycle_columns'][k]
            value -= fun(row[i_m_reference], beta)
        return value
    return interf_corr_row


def create_interf_corrfuncs(
        file_spec: Spec
        ) -> List[Callable[
            [IntensRow, float],
            float]]:
    '''
    Helper function for get_interf_corr.
    '''
    cols = file_spec['cycle_columns']
    interf = file_spec['used_isotopes']
    funcs = []
    for column, i in cols.items():
        if column in interf:
            funcs.append(create_interf_corr(column, file_spec))
        else:
            funcs.append(create_row_lookup(i))
    return funcs


def get_interf_corr(
        file_spec: Spec,
        ) -> Callable[[IntensRow, float], IntensRow]:
    '''
    Generate lookups and interference correction functions
    for all masses. Based on "used_isotopes", "masses", and
    "nat_ratios" in file_spec.
    Returns a function which takes a data row and mass bias factor
    and returns an interference corrected row.
    '''
    funcs = create_interf_corrfuncs(file_spec)

    def corr_row(
                row: IntensRow,
                beta: float,
                ) -> IntensRow:
        return np.array([f(row, beta) for f in funcs])
    return corr_row


def get_interf_corr_fun(
        file_spec: Spec
        ) -> Callable[
            [IntensRow, float],
            Tuple[IntensRow, RatioRow]]:
    '''
    Returns a function wich takes a row and mass bias factor as
    arguments, and returns interfere corrected intensities and
    interference corrected ratios for data inversion, specified in
    "reduce_fracs"
    '''

    corr_fun = get_interf_corr(file_spec)
    column_indices = file_spec['cycle_columns']
    m_ref, m_others = file_spec['reduce_fracs']
    corrected_masses = [i for c, i in column_indices.items()
                        if c in file_spec['used_isotopes'].keys()]
    i_ref = column_indices[m_ref]
    i_others = [column_indices[m] for m in m_others]

    def get_interf_corr_vals(
                row: IntensRow,
                beta: float,
                ) -> Tuple[IntensRow, RatioRow]:
        iexpcorr = corr_fun(row, beta)
        val_ref = iexpcorr[i_ref]
        interfexpcorr_intensities = [iexpcorr[i] for i in corrected_masses]
        interfexpcorr_ratios = [iexpcorr[i]/val_ref for i in i_others]

        return interfexpcorr_intensities, interfexpcorr_ratios

    return get_interf_corr_vals


def get_reduction_fun(
        file_spec: Spec,
        ) -> Callable[
            [RowMx, List[int]],
            RowMx]:
    """
    Generates the data inversion function
        "calculate_reduction(cycles, skip_rows=[])"
    Which takes all cycles, and reduces them. Returns a list of rows.
    """

    iexpcorr = get_interf_corr_fun(file_spec)
    alpha_beta_lambda_0 = np.array(file_spec['initial_parameters'])

    elem = file_spec['element']
    red_denom, red_numerats = file_spec['reduce_fracs']

    masses = file_spec['masses']
    denom_mass = masses['{0}{1}'.format(red_denom, elem)]
    mass_ratios = np.array([
        masses['{0}{1}'.format(red_numer, elem)]/denom_mass
        for red_numer in red_numerats
    ])

    Pi_values = np.log(mass_ratios)

    spk_ratios = np.array([
        file_spec['spike']['{1}{2}/{0}{2}'.format(
            red_denom, red_numer, elem)]
        for red_numer in red_numerats
    ])

    std_ratios = np.array([
        file_spec['standard']['{1}{2}/{0}{2}'.format(
            red_denom, red_numer, elem,)]
        for red_numer in red_numerats
    ])

    def gen_fn(
                meas_rat: RowMx,
                ) -> Callable[[RowMx], float]:
        '''
        Generate cost function for gradient descent method.
        '''

        def fn(
                    alpha_beta_lambda: AlphaBetaLambda,
                    ) -> float:
            alpha, beta, lbda = alpha_beta_lambda
            q = spk_ratios * lbda
            # Refactor to use exp_corr function instead of explicit calculation
            with np.errstate(invalid='raise'):
                try:
                    p = np.exp(np.log(meas_rat) - Pi_values * beta)
                    r = np.exp(np.log(std_ratios * (1-lbda)) - Pi_values * alpha)
                except FloatingPointError:
                    p = np.array(np.nan)
                    r = np.array(np.nan)
            return q - p + r
        return fn
    
    def calculate_reduction(
                cycles: RowMx,
                skip_rows: List[int] = [],
                ) -> RowMx:
        results = []

        for i, row in enumerate(cycles):
            if i+1 in skip_rows:
                continue
            beta = 0
            alpha_beta_lambda = np.array(alpha_beta_lambda_0)
            for _ in range(2):
                interfexpcorr_intensities, interfexpcorr_ratios = iexpcorr(row, beta)
                
                opt_fn = gen_fn(interfexpcorr_ratios)

                with warnings.catch_warnings(): 
                    warnings.simplefilter("error", RuntimeWarning) 
                    try:
                        alpha_beta_lambda = opt.fsolve(opt_fn, alpha_beta_lambda_0)
                    except RuntimeWarning:
                        alpha_beta_lambda = np.array([0,0,0])
                beta = alpha_beta_lambda[1]
            results.append(
                [*row, *alpha_beta_lambda, *interfexpcorr_intensities])
        return results

    return calculate_reduction


def generate_raw_labels(
        file_spec: Spec,
        ) -> List[str]:
    '''
    Generates data labels for values returned by the 
    "calculate_reduction" function generated by "get_reduction_fun"
    '''
    elem = file_spec['element']
    labels = ['raw_' + v for _, v
              in sorted([(i, v) for v, i
                         in file_spec['cycle_columns'].items()])]
    labels += ['alpha_nat', 'beta_ins', 'lambda']
    labels += ['meas_{}{}'.format(k, elem) for k in sorted(file_spec['used_isotopes'].keys())]
    return labels


def invert_data(
        cycles: RowMx,
        file_spec: Spec,
        columns: bool = False
        ) -> Tuple[List[str], Dict[str, Column]]:
    '''
    Data inversion function.
    returns labels, and data in row major order.
    '''

    # Transpose data if provided in columns
    if columns:
        cycles = [*zip(*cycles)]

    cal_red = get_reduction_fun(file_spec)
    labels = generate_raw_labels(file_spec)
    masses = file_spec['masses']
    standard = file_spec['standard']
    spike = file_spec['spike']
    elem = file_spec['element']
    reduced = cal_red(cycles)
    interferences = {k: v for k, v in file_spec['used_isotopes'].items() if len(v) > 0}
    nat_ratios = file_spec['nat_ratios']

    results = OrderedDict()
    for k, c in zip(labels, zip(*reduced)):
        results[k] = np.array(c)

    # Get denominator value for ratios
    den, numerators = file_spec['report_fracs']
    den_col = results['meas_{}{}'.format(den, elem)]
    den_str = '{}{}'.format(den, elem)

    # Calculate measured interference corrected ratios
    for num in numerators:
        num_str = '{}{}'.format(num, elem)
        ratio_lab = 'meas_{}/{}'.format(num_str, den_str)
        labels.append(ratio_lab)
        try:
            results[ratio_lab] = (results['meas_{}'.format(num_str)]/den_col)
        except FloatingPointError:
            results[ratio_lab] = np.array(np.nan)

    # Calculate interference ratios
    for den, interfs in interferences.items():
        # interf_den_str = '{}{}'.format(den, 'raw')
        interf_den_col = results['raw_{}'.format(den)]
        for num, interf_elem in interfs:
            nat_rat = nat_ratios['{0}{2}/{1}{2}'.format(num, den, interf_elem)]
            num_str = '{}{}'.format(num, interf_elem)
            interf_num_col = results['raw_{}'.format(num)]
            ratio_lab = 'interf_{}{}_ppm'.format(den, interf_elem)
            labels.append(ratio_lab)
            results[ratio_lab] = 1e6*(interf_num_col/interf_den_col)/nat_rat

    # Calculate solution ratios, instrument fract. corrected.
    for num in numerators:
        num_str = '{}{}'.format(num, elem)
        ratio_lab = 'soln_{}/{}'.format(num_str, den_str)
        labels.append(ratio_lab)
        init_rat = results['meas_{}/{}'.format(num_str, den_str)]
        mass_rat = masses[num_str]/masses[den_str]
        results[ratio_lab] = exp_corr(init_rat, mass_rat, results['beta_ins'])

    # Calculate sample ratios, natural fract corrected.
    for num in numerators:
        num_str = '{}{}'.format(num, elem)
        ratio_lab = 'sample_{}/{}'.format(num_str, den_str)
        labels.append(ratio_lab)
        init_rat = standard['{}/{}'.format(num_str, den_str)]
        mass_rat = masses[num_str]/masses[den_str]
        results[ratio_lab] = exp_corr(init_rat, mass_rat, results['alpha_nat'])

    # Calculate delta values
    rel_labels = []
    for rel_lab, (num_str, den_str, factor, _) in file_spec['rel_report'].items():
        rel_labels.append(rel_lab)
        std = standard['{}/{}'.format(num_str,den_str)]
        spl = results['sample_{}/{}'.format(num_str,den_str)]
        results[rel_lab] = ((spl/std)-1)*factor
        results.move_to_end(rel_lab, last=False)
    labels = rel_labels + labels

    # Calculate Q-value (i.e. Q_52Cr_54Cr for 52Cr spiked with 54Cr ratios)
    rep_iso = file_spec['report_fracs'][0]
    spk_iso = file_spec['reduce_fracs'][0]
    Q_label = 'Q_{0}{2}/{1}{2}'.format(spk_iso, rep_iso, elem)
    labels.append(Q_label)
    results[Q_label] = (
        (results['soln_{0}{2}/{1}{2}'.format(spk_iso, rep_iso, elem)] -
         file_spec['spike']['{0}{2}/{1}{2}'.format(spk_iso, rep_iso, elem)])/
        (results['sample_{0}{2}/{1}{2}'.format(spk_iso, rep_iso, elem)] - 
         results['soln_{0}{2}/{1}{2}'.format(spk_iso, rep_iso, elem)]
        ))

    # Calculate conc factor F_conc
    # C_spl = F_conc * wt_spk * C_spk / wt_spl
    # F_conc = Q * rep_iso_abund_spk * mol_mass_spl / (rep_iso_abund_spl * mol_mass_spk)
    labels.append('F_conc')
    other_iso = file_spec['report_fracs'][1]
    all_iso = [rep_iso] + other_iso
    ratios = [results['sample_{1}{2}/{0}{2}'.format(rep_iso, num_iso, elem)] for num_iso in other_iso]
    abund_rep_spl = 1/(sum(ratios) + 1)
    all_abund =[abund_rep_spl] + [r*abund_rep_spl for r in ratios]
    mol_mass_spl = sum([
        abund * masses['{}{}'.format(mass_label,elem)]
        for abund, mass_label in zip(all_abund, all_iso)])
    abund_rep_spk = spike['{}{}'.format(rep_iso,elem)]
    mol_mass_spk = spike['amu']
    results['F_conc'] = results[Q_label] * abund_rep_spk * mol_mass_spl / (abund_rep_spl * mol_mass_spk)

    return results


def gen_filter_function(
        iqr_limit: float,
        max_fraction: float
        ) -> Callable[[Column, bool], Column]:
    def filter_data(
            data: Column,
            get_rejected_cycle_idx: bool = False
            ) -> Column:
        '''
        Outlier rejection for satistics calculation.
        '''
        mean = np.median(data)
        limit = iqr_limit * stats.iqr(data)
        N = len(data)
        max_num = ceil(N*max_fraction)

        data = sorted(data, key=lambda v: abs(v-np.mean(data)), reverse=True)
        filtered = []
        rejected = []
        n = 0
        for i, v in enumerate(data):
            if (n >= max_num):
                filtered.append(v)
            elif (abs(v-mean) <= limit):
                filtered.append(v)
            else:
                n += 1
                rejected.append(i)
        if get_rejected_cycle_idx:
            result = np.array(rejected)
        else:
            result = np.array(filtered)
        return result
    return filter_data


def calc_stat(
        data_in: Column,
        label: str,
        mean: bool = True,
        std: bool = False,
        rsd: bool = False,
        se: bool = False,
        minim: bool = False,
        maxim: bool = False,
        N: bool = False,
        ) -> Tuple[List[str], Column]:
    '''
    Helper function for caluclating summary statistics.
    '''
    data_out = []
    labels_out = []
    if mean:
        labels_out.append(label)
        data_out.append(data_in.mean())
    if std:
        labels_out.append(label+'_2STD')
        data_out.append(2*data_in.std())
    if rsd:
        labels_out.append(label+'_2RSD_ppm')
        data_out.append(2e6*data_in.std()/data_out[label])
    if se:
        labels_out.append(label+'_2SE')
        data_out.append(2*data_in.std()/(len(data_in)**0.5))
    if minim:
        labels_out.append(label+'_min')
        data_out.append(data_in.min())
    if maxim:
        labels_out.append(label+'_max')
        data_out.append(data_in.max())
    if N:
        labels_out.append(label+'_N')
        data_out.append(len(data_in))
    return labels_out, data_out


def summarise_data(
        results: Dict[str, List[float]],
        file_spec: Spec,
        ) -> Dict[str, float]:
    '''
    Calculate sample statistics.
    Generates labels and a single row/series of summary data.
    '''

    summary = OrderedDict()
    rel_report = file_spec['rel_report']
    filter_data = gen_filter_function(**file_spec['outlier_rejection'])

    for label in results.keys():
        if label in rel_report:
            filtered = filter_data(results[label])
            slabs, svals = calc_stat(
                filtered, label,
                mean=True, std=True, rsd=False, se=True,
                minim=False, maxim=False, N=True)
            for slab, val in zip(slabs, svals):
                summary[slab] = val
            _, _, _, unfiltered = rel_report[label]
            if unfiltered:
                slabs, svals = calc_stat(
                    results[label], label + '_unfiltered',
                    mean=True, std=True, rsd=False, se=True,
                    minim=False, maxim=False, N=True)
                for slab, val in zip(slabs, svals):
                    summary[slab] = val
        elif label == 'raw_{}'.format(file_spec['report_fracs'][0]):
            slabs, svals = calc_stat(
                results[label], label,
                mean=True, std=True, rsd=False, se=False,
                minim=True, maxim=True, N=True)
            for slab, val in zip(slabs, svals):
                summary[slab] = val
        elif 'raw' in label:
            summary[label] = results[label].mean()
        elif label == 'beta_ins':
            slabs, svals = calc_stat(
                results[label], label,
                mean=True, std=True, rsd=False, se=False,
                minim=False, maxim=False, N=False)
            for slab, val in zip(slabs, svals):
                summary[slab] = val
        else:
            summary[label] = results[label].mean()

    return summary
