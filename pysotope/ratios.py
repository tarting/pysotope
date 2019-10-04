'''
Helper functions to calculate isotope ratios
'''

import re
from collections import OrderedDict

import click
import numpy as np

from pysotope.typedefs import (
        List,
        Dict,
        Iterable,
        Union,
        Spec,
        )


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


def calc_abund(
        ratios: List[float],
        ratios_key: str,
        file_spec: Spec,
        ) -> Dict[str, float]:
    '''
    Calculate isotope abundance from ratios
    '''
    elem = file_spec['element']
    denom, numerators = file_spec[ratios_key]
    abund_denom = 1/(sum(ratios) + 1)

    abund = dict()
    abund[denom + elem] = abund_denom
    for i, numer in enumerate(numerators):
        abund[numer + elem] = ratios[i] * abund_denom

    return abund


def calc_spec_abund(
        composition_key: str,
        ratios_key: str,
        file_spec: Spec,
        ) -> Dict[str, float]:
    '''
    Calculate isotope abundance from ratios
    '''
    elem = file_spec['element']
    composition = file_spec[composition_key]
    denom, numerators = file_spec[ratios_key]
    denom_key = denom + elem
    ratios = []
    for numer in numerators:
        rat_key = '{}{}/{}'.format(numer, elem, denom_key)
        ratios.append(composition[rat_key])

    abund = calc_abund(ratios, ratios_key, file_spec)
    if 'name' in composition:
        abund['name'] = composition['name']
    else:
        abund['name'] = composition_key

    return abund


def mix_abund(
        abund_1: Dict[str, float],
        abund_2: Dict[str, float],
        fraction_1: float,
        ) -> Dict[str, float]:
    '''
    Mix two endmember compositions.
    Useful for testing, and calibration.
    '''
    mixture = dict()

    for k in abund_1.keys():
        if k == 'name':
            continue
        else:
            val_1 = abund_1[k] * fraction_1
            val_2 = abund_2[k] * (1 - fraction_1)
            mixture[k] = val_1 + val_2

    return mixture


def row_from_abund(
        abund: Dict[str, float],
        file_spec: Spec,
        interferences: Dict[int, float] = None,
        ) -> Iterable[float]:
    '''
    Generate an artificial analysis cycle row
    '''
    indices = file_spec['cycle_columns']
    row = [0 for _ in indices]
    for isotope, value in abund.items():
        if '/' not in isotope:
            try:
                mass = re.findall('[0-9]+', isotope)[0]
                row[indices[mass]] += value
            except IndexError:
                pass
            except KeyError:
                pass
    if interferences:
        for mass, value in interferences.items():
            if mass in indices:
                row[indices[mass]] += value

    return row


def calc_amu(
        abund: dict,
        file_spec: Spec,
        ) -> float:
    '''
    Calculate the relative atomic weight of all isotopes/.
    '''
    masses = file_spec['masses']
    amu = sum([v * masses[k]
               for k, v in abund.items()
               if k != 'name'])
    return amu


def generate_ratios(
        abund: Dict[str, float],
        file_spec: Spec
        ) -> Dict[str, float]:
    '''
    Generate all possible ratios and relative atomic weight
    for an isotope composition given as relative abundances.
    '''
    comp = OrderedDict(abund)
    comp['amu'] = calc_amu(abund, file_spec)
    for denom in abund.keys():
        if denom == 'name':
            comp['name'] = abund['name']
        else:
            comp.update(
                {'{}/{}'.format(k, denom): v / abund[denom]
                 for k, v in abund.items()}
            )
    return comp


def generate_isotope_labels(
        element: str,
        ratio_spec: List[Union[str, List[str]]],
        ) -> List[str]:
    '''
    Generate isotope labels from element and ratio specification list.
    '''
    numerators = list(sorted([ratio_spec[0]] + ratio_spec[1]))
    return [n + element for n in numerators]


def calc_abund_ratio(
        abund: Dict[str, float],
        numer_key: str,
        denom_key: str,
        file_spec: Spec,
        ) -> float:
    try:
        numer_val = abund[numer_key]
        denom_val = abund[denom_key]
    except KeyError as msg:
        click.echo(
            'ERROR  | spec_file: Key missing for {} or {} in {}\n{}'.format(
                denom_key, numer_key, reference_key, msg),
            err=True)
        raise
    return numer_val/denom_val


def calc_one_ratio(
        reference_key: str,
        numer_key: str,
        denom_key: str,
        file_spec: Spec,
        ) -> float:
    '''
    Get a single isotope ratio from the file_spec defined samples e.g. spike
    and standard). numer_key and denom_key must be full isotope name e.g. 50Cr
    '''
    abund = file_spec[reference_key]
    return calc_abund_ratio(abund, numer_key, denom_key, file_spec)


def calc_abund_ratios(
        abund: Dict[str, float],
        denom: str,
        numerators: List[str],
        file_spec: Spec,
        ) -> Iterable:
    ratios = np.ones(len(numerators))
    denom_key = '{}{}'.format(denom, file_spec['element'])
    for i, numer in enumerate(numerators):
        numer_key = '{}{}'.format(numer, file_spec['element'])
        ratio_val = calc_abund_ratio(
            abund, numer_key, denom_key, file_spec)
        ratios[i] *= ratio_val

    return ratios


def calc_spec_ratios(
        reference_key: str,
        ratio_key: str,
        file_spec: Spec,
        ) -> Iterable:
    '''
    Calculate ratios for a given denominator numerator set from file_spec
    compositions spike and standard.
    '''
    abund = file_spec[reference_key]
    denom, numerators = file_spec[ratio_key]
    return calc_abund_ratios(abund, denom, numerators, file_spec)


def fractionate_abund(
        abund: Dict[str, float],
        ratio_key: str,
        frac_fact: float,
        file_spec: Spec,
        ) -> Iterable:
    elem = file_spec['element']
    denom, numerators = file_spec[ratio_key]
    masses = file_spec['masses']
    ratios = calc_abund_ratios(abund, denom, numerators, file_spec)
    denom_key = denom + elem
    new_ratios = [0 for _ in ratios]
    for i, numer in enumerate(numerators):
        numer_key = numer + elem
        mass_ratio = masses[numer_key] / masses[denom_key]
        frac_rat = exp_corr(ratios[i], mass_ratio, frac_fact)
        new_ratios[i] = frac_rat
    
    fract_abund = calc_abund(new_ratios, ratio_key, file_spec)
    return fract_abund

