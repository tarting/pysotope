'''
Helper functions to calculate isotope ratios
'''

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
    try:
        numer_val = file_spec[reference_key][numer_key]
        denom_val = file_spec[reference_key][denom_key]
    except KeyError as msg:
        click.echo(
            'ERROR  | spec_file: Key missing for {} or {} in {}\n{}'.format(
                denom_key, numer_key, reference_key, msg),
            err=True)
        raise
    return numer_val/denom_val


def calc_spec_ratios(
        reference_key: str,
        ratio_key: str,
        file_spec: Spec,
        ) -> Iterable:
    '''
    Calculate ratios for a given denominator numerator set from file_spec
    compositions spike and standard.
    '''
    denom, numerators = file_spec[ratio_key]
    ratios = np.ones(len(numerators))
    denom_key = '{}{}'.format(denom, file_spec['element'])
    for i, numer in enumerate(numerators):
        numer_key = '{}{}'.format(numer, file_spec['element'])
        ratio_val = calc_one_ratio(
            reference_key, numer_key, denom_key, file_spec)
        ratios[i] *= ratio_val

    return ratios
