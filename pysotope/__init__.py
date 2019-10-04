'''
pysotope - a package for inverting double spike isotope analysis data
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


name = 'pysotope'


import os
import sys
from datetime import datetime as dt
from collections import namedtuple, OrderedDict

import click

import pysotope.data_reader as data_reader
import pysotope.diagrams as diagrams
import pysotope.exceptions as exceptions
import pysotope.filelist as filelist
import pysotope.invert as invert
import pysotope.ratios as ratios
import pysotope.run as run
import pysotope.typedefs as typedefs

from pysotope.data_reader import read_xls, read_spec_file, parse_date
from pysotope.invert import invert_data, summarise_data, exp_corr
from pysotope.diagrams import generate_cycleplots, generate_summaryplot
from pysotope.run import reduce_data
from pysotope.typedefs import (
        Data, Spec, Dict, Any, List, Callable, Tuple,)

#labeled = namedtuple('labeled', ['labels', 'data'])
#reduced = namedtuple('reduced', ['summary', 'data'])



def get_xls_inverter_from_spec(
        file_spec: Spec, 
        ) -> Callable[[str], Data]:
    '''
    Get a data reducer from spec file.
    '''
    def invert_xls(
            file_path: str,
            ) -> Data:
        '''
        Reduce a file directly from xls
        '''
        data = read_xls(file_path, file_spec)
        reduced = invert_data(data['CYCLES'], file_spec)

        summary = summarise_data(cycles, file_spec)
        
        summary['file_path'] = file_path
        summary.move_to_end('file_path', last=False)
        
        summary_labels.append('file_spec_path')
        if 'file_spec_path' in file_spec:
            summary['file_spec_path'] = file_spec['file_spec_path']
        else: 
            summary['file_spec_path'] = 'N/A'

        summary['proc_time'] = dt.now().strftime(
                file_spec['date']['report_format'])
        
        data['REDUCED'] = reduced
        data['SUMMARY'] = summary

        return data
    return invert_xls

def get_xls_inverter(
        file_spec_path: str,
        ) -> Callable[[str], Data]:
    '''
    Get xls inverter from file spec path
    '''
    file_spec = read_spec(file_spec_path)
    file_spec['file_spec_path'] = file_spec_path
    return get_xls_inverter_from_spec(file_spec)

def invert_from_paths(
        xls_path: str,
        file_spec_path: str,
        ) -> Data:
    '''
    Invert directly from xls and file spec path.
    '''
    invert_xls = get_xls_inverter(file_spec_path)
    reduced = invert_xls(xls_path)
    return reduced


def safe_read_file(
        filepath: str, 
        spec: Spec,
        ) -> Tuple[str, Data]:
    '''
    Read file while catching errors
    Returns empty dict and prints warning for all other errors.
    '''
    filename = os.path.split(filepath)[1][:-4]
    try: 
        data = read_xls(filepath, spec)
    except Exception as e:
        sys.stdout.flush()
        click.echo('\rERROR  | while reading file {}: {}'.format(
					click.format_filename(filepath), e), 
              err=True)
        data = OrderedDict()
    return filename, data

def safe_invert_data(
        data: Data,
        spec: Spec,
        ) -> Data:
    '''
    invert_data wrapped in try block.
    Returns empty dict for empty supplied data and empty dict and prints 
    warning for all other errors.
    '''
    if data: # Check if empty
        try:
            reduced = invert_data(data['CYCLES'], spec)
        except Exception as e:
            sys.stdout.flush()
            click.echo('\rERROR  | while reducing file: {}'.format(e), 
                  err=True)
            reduced = OrderedDict()
    else:
        reduced = OrderedDict()
    return reduced

def safe_summarise_data(
        reduced: Data,
        spec: Spec,
        ) -> Data:
    '''
    Summarise data wrapped in try block
    Returns empty dict for empty supplied data and empty dict and prints 
    warning for all other errors.
    '''
    try: 
        summary = summarise_data(reduced, spec)
    except Exception as e:
        sys.stdout.flush()
        click.echo('\rERROR  | while summarising file: {}'.format(e), 
              err=True)
        summary = OrderedDict()
    return summary

