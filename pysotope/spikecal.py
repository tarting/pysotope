'''With
'''

import os
import re
import copy
import json
import pickle
from collections import OrderedDict
from contextlib import contextmanager

import pandas as pd
from tqdm import tqdm
from scipy import optimize as opt

import pysotope.ratios as ratios
from pysotope.data_reader import read_spec_file, DataReader
from pysotope.invert import invert_data


@contextmanager
def __cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def read_from_extvar(
        extvar_path: str,
        spec_path: str,
        ) -> dict:
    '''Performs IO'''
    extvars = pd.read_excel(extvar_path, index_col=0)

    dir_path, _ = os.path.split(extvar_path)
    spec = read_spec_file(spec_path)

    runs = OrderedDict()

    with __cwd(dir_path):
        for file_id, row in tqdm(
                    extvars.iterrows(),
                    total=len(extvars),
                    ):
            if row['ignore']:
                pass
            else:
                file_reader = DataReader(spec)
                raw_data = file_reader(row['filepath'], spec)
                raw_data['metadata'] = OrderedDict(row)
                runs[file_id] = raw_data

    return runs

def read_data(
            ext_vars_path: str,
            spec_path: str,
            reload: str = False,
            pickle_file: str = 'cal_standards.db'
            ) -> dict:
    if not os.path.isfile(pickle_file):
        reload = True

    if reload:
        runs = read_from_extvar(
            ext_vars_path,
            spec_path,
        )
        with open(pickle_file, 'wb') as fh:
            pickle.dump(runs, fh)
    else:
        with open(pickle_file, 'rb') as fh:
            runs = pickle.load(fh)
    return runs

def cycle_sum(
            raw: dict,
            metadata_key:
            str='metadata'
            ) -> dict:
    '''Pure function?'''
    cycles = raw['CYCLES']

    if metadata_key in raw:
        try:
            first_row = raw[metadata_key]['first_row']
        except KeyError:
            first_row = 0
        try:
            last_row = raw[metadata_key]['last_row']
        except KeyError:
            last_row = len(cycles)

    sum_ = [sum(col) for col in zip(*cycles)]
    return sum_

def get_sums(runs):

    sample_ids = []
    sum_table = []
    for sample_id, run in runs.items():
        sample_ids.append(sample_id)
        sum_ = cycle_sum(run)
        sum_table.append(sum_)

    return sample_ids, sum_table

def inversion_summary(
            sum_table: list,
            spec: dict,
            weighted: bool=True
            ) -> (float, float, float):
    inverted = invert_data(sum_table, spec)

    param_label, rel_report = next(
        iter(spec['rel_report'].items()))
    parameter = inverted[param_label]
    if weighted:
        weights = inverted['meas_' + rel_report[1]]
        weights_i = weights / weights.sum()
        mean = (parameter * weights_i).sum()
        stdev = ((weights_i * (parameter-mean)**2).sum()/
                         ((len(weights_i - 1)* sum(weights_i))
                          /len(weights_i))
                         )**0.5
        sterr = (stdev**2 * weights**2).sum()**0.5 / (weights).sum()

    else:
        mean = parameter.mean()
        stdev = parameter.std()
        sterr = stdev / (len(parameter)-1)**0.5

    return mean, stdev, sterr

def generate_cost_function(cycles, spec):
        global cycle_counter
        cycle_counter = 0
        opt_spec = copy.deepcopy(spec)
        spike_meas = dict(spec['spike'])
        def fn(alpha_spike):
            global cycle_counter
            cycle_counter += 1

            corr_spike = ratios.fractionate_abund(spike_meas, 'report_fracs', alpha_spike, spec)
            opt_spec['spike'] = corr_spike
            err, _, _ = inversion_summary(cycles, opt_spec)

            return err
        return fn

def optimize_spike(
            cycles: list,
            spec: dict,
            alpha_0: float = 0
            ) -> float:
    _, (numer, denom, _, _) = next(iter(spec['rel_report'].items()))

    f_cost = generate_cost_function(cycles, spec)
    alpha_opt = opt.fsolve(f_cost, alpha_0)
    return alpha_opt[0]

def optimize_spec(
            ext_vars_path: str,
            spec_path: str,
            reload: str = False,
            pickle_file: str = 'cal_standards.db',
            alpha_0: float = 0,
            ) -> dict:
    spec = read_spec_file(spec_path)
    runs = read_data(
            ext_vars_path, spec_path,
            reload, pickle_file)
    run_ids, sums = get_sums(runs)

    alpha_corr = optimize_spike(sums, spec, alpha_0=alpha_0)

    _, (numer, denom, _, _) = next(
        iter(spec['rel_report'].items()))

    new_spike = ratios.fractionate_abund(
        spec['spike'],
        'report_fracs',
        alpha_corr,
        spec,
    )


    spike = spec['spike']
    spike.update(new_spike)
    spike['MB_corr_type'] = '{}/{}'.format(numer, denom)
    spike['alpha_calibration'] = alpha_corr

    rel_label, _ = next(
        iter(spec['rel_report'].items()))
    rel_mean, rel_std, rel_se = inversion_summary(sums, spec)

    spike['{}_mean_weighted'.format(rel_label)] = rel_mean
    spike['{}_2STD'.format(rel_label)] = 2*rel_std
    spike['{}_2SE'.format(rel_label)] = 2*rel_se
    spike['{}_N'.format(rel_label)] = len(sums)

    inverted = invert_data(sums, spec)

    return spec


if __name__ == '__main__':
    print('Double spike isotope calibration utility')
