'''
Filetype plugin for .exp files for the Thermo-Fisher Neptune instrument at
Rutgers University Geochemistry Lab.
'''

# pysotope - a package for inverting double spike isotope analysis data
#
#     Copyright (C) 2019 Trygvi Bech Arting, University of Copenhagen
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

import os
import csv
from datetime import datetime as dt
from dateutil import parser as dtparser

import numpy as np

from pysotope.typedefs import Spec, Data, Any, List, Dict, Union


def exp_dump(
        in_file: str,
        out_file: str = None,
        ) -> str:
    '''
    Dumps the exp data to a string or file
    '''
    if os.path.isfile(in_file):
        in_path = os.path.abspath(in_file)
        with open(in_path, 'r') as fh:
            output = fh.read()
        if out_file is not None:
            with open(out_file, 'w') as file_handle:
                file_handle.write(output)
    else:
        raise FileNotFoundError('Input file not found', in_file)

    return output



def is_in_row(
            row: list,
            string: str
            ) -> bool:
    in_row = False
    for entry in row:
        if string in entry:
            in_row = True
            break
    return in_row


def read_table(rows, table_def, header_def):
    in_table = False
    row_no = 0

    # Table data
    row_len = table_def['n_columns']
    row_fst = table_def['first_col']
    row_end = row_fst + row_len
    end_string = table_def['end_string']
    start_string = table_def['start_string']

    # Header data
    header_fst = header_def['first_col']
    header_end = header_fst + header_def['n_columns']
    cycles_headers = [str(i) for i in range(header_def['n_columns'])]
    cycles = []
    for i, row in enumerate(rows):

        if is_in_row(row, header_def['search_string']):
            cycles_headers = row[header_fst: header_end]

        if in_table and is_in_row(row, end_string):
            in_table = False
        elif is_in_row(row, start_string):
            in_table = True

        if in_table and (row_no < table_def['skip_rows']):
            row_no += 1
        elif in_table:
            row_data = row[row_fst: row_end]

            try:
                cycle = [float(s) for s in row_data]
            except ValueError as error:
                print('Cannot parse field in line {}\n{}\nRaises: {}'.format(
                        i, row, error), file=sys.stderr)
                row_no += 1
                continue




            cycles.append(cycle)

            row_no += 1

    cycles_n = len(cycles)

    return cycles, cycles_headers, cycles_n


def get_param_val(row, labels):
    entry = row[0]
    for label in labels:
        if label in entry:
            break
    return label, entry

def read_params(rows, param_def):
    labels = param_def['labels']

    data = dict()
    for row in rows:
        label, value = get_param_val(row, labels)
        data[label] = value[len(label)+2:]
    return data


################################################################################
# Reader function
################################################################################
def read_file(
        file_path: str,
        file_spec: Spec,
        ) -> Data:
    '''
    Parses a exp document at file_path into a python dictionary.
    Given a file_spec in dict format.
    '''

    raw = exp_dump(file_path)
    lines = raw.splitlines()
    rows = [row.split('\t') for row in lines]

    data = dict()
    for label, data_spec in file_spec['file_spec'].items():
        print(label, data_spec)
        data_type = data_spec[0]
        data_spec = data_spec[1:]

        if data_type == 'table':
            table_def, header_def = data_spec
            t_data, t_columns, t_n = read_table(
                    rows,
                    table_def,
                    header_def,
            )
            data[label] = t_data
            data[label+'_columns'] = t_columns
            data[label+'_N'] = t_n
        elif data_type == 'params':
            param_def = data_spec[0]
            params = read_params(rows, param_def)
            data[label] = params

    return data

FILETYPE_EXTENSION = '.exp'
