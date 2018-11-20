'''
pysotope - a package for inverting double spike isotope analysis data
This module is specific to processing xls datafiles produced by Isotop-X mass spectrometers.
The csv parsing module can be used to process any csv file (currently only comma separated values
and with decimal dots). The xls_dump function reads any xls document given a xls to csv converter
given by the converer_path, which dumps the contents of sheets contiguously to stdout.
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


import os
import sys
import subprocess
import json
from datetime import datetime as dt
from functools import reduce

import numpy as np

from pysotope.exceptions import UndefinedDataDefinition, UnknownPlatform

# Move to settings module, only used in the read_xls function
# path to a converter which dumps the contents of an xls to stdout
# as csv, appending all worksheets.
# Currently using binary shipped with the Haskell xls module.
if sys.platform == 'win32':
    converter = 'xls2csv.exe'
elif sys.platform == 'darwin':
    converter = 'xls2csv_macos_x86-64' 
elif sys.platform == 'linux':
    converter = 'xls2csv'
else:
    raise UnknownPlatform('Platform [{}] unknown'.format(sys.platform))
converter_path = os.path.join(sys.prefix, 'pysotope','bin', converter) #os.path.abspath(converter_path)
print(converter_path)

# HACK: Global current file variable is set and unset in the read_xls function (now procedure)
current_xls_file = ''
current_row = -1

def read_json(file_path):
    with open(file_path, 'r') as fh:
        file_spec = json.load(fh)
    return file_spec


def xls_dump(converter_path, in_file, sheet_index=None, out_file=None):
    '''
    Dumps the content of an xls document into a contiguous csv string.
    Optionally dumps the string into a UTF-8 formatted file.
    '''
    if os.path.isfile(in_file):
        in_path = os.path.abspath(in_file)
        if sheet_index is None:
            output = subprocess.check_output([converter_path ,in_path], universal_newlines=True)
        else:
            output = subprocess.check_output([converter_path ,in_path, sheet_index], universal_newlines=True)
        if out_file is not None:
            with open(out_file,'w') as fh:
                fh.write(output)
    else:
        raise FileNotFoundError('Input file not found',in_file)
    
    return output

def parse_float(v):
    try:
        value = float(v)
    except ValueError as e:
        print('Could not parse float {} in {} row {}'.format(v, current_xls_file, current_row), e, sep='\n', file=sys.stderr )
        value = np.nan
    return value


def get_table(contents, start_string, end_string, first_col, n_columns, skip_rows=1):
    '''
    Read a data table by reading keywords, gets set number of columns. 
    Skips row with keyword by default.
    '''
    global current_row
    start = contents[:contents.find(start_string)].rfind('\n') + 1
    table = contents[start:]
    table = table[:table.find(end_string)]
    table = table.splitlines()
    table = [l.split(',') for l in table]
    table = [l for l in table if len(reduce(lambda a, b: a+b, l)) > 0]
    table = [l[first_col:first_col+n_columns] for l in table[skip_rows:]]
    n = len(table[0])
    float_table = []
    skipped_rows = 0
    for i, r in enumerate(table):
        current_row = i - skipped_rows
        if len(r) == n:
            try:
                row = [float(v) for v in r]
                float_table.append(row)
            except ValueError as e:
                print(e, file=sys.stderr)
                print('Could not parse value in {} row {}: {}'.format(current_xls_file, current_row, r), e, sep='\n', file=sys.stderr )
                #float_table.append([np.nan for _ in range(n)])
                skipped_rows += 1
        else:
            skipped_rows += 1
    current_row = -1
#    table = [[float(v) for v in r] for r in table]
    return float_table


def get_keyword_row(contents, search_string, first_col, n_columns):
    '''
    Read a row containing a keyword. For fetching headers.
    '''
    start = contents[:contents.find(search_string)].rfind('\n') + 1
    line = contents[start:]
    line = line[:line.find('\n')]
    line = line.split(',')
    return line[first_col:first_col+n_columns]


def get_params(contents, labels, start_string):
    '''
    Read value following comma after keyword.
    '''
    ctrl = contents[contents.find(start_string):]
    results = []
    for l in labels:
        res = ctrl[ctrl.find(l):]
        res = res[:res.find('\n')].split(',')[1]
        results.append(res)
    return results


def read_spec(contents, spec):
    '''
    Parse a single table or parameter specification in a file_spec document. 
    '''
    spec_type = spec[0]
    if spec_type == 'table':
        data = [get_keyword_row(contents, **spec[2])]
        data += get_table(contents, **spec[1])

    elif spec_type == 'params':
        data = get_params(contents, **spec[1])
        labels = spec[1]['labels']
        data = zip(labels, data)
    else:
        raise UndefinedDataDefinition('Unknown data definition: {}'.format(spec_type))
    return data


def distill_to_csv(contents, file_spec):
    '''Takes xls csv dump and extracts data according to the given file specifications, returns a string'''
    aggregated = str()
    for k, spec in file_spec.items():
        aggregated += k + '\n'
        data = read_spec(contents, spec)
        for r in data:
            aggregated += ','.join(r) + '\n'
    return aggregated


def parse_date(data, file_spec, as_datetime_object=False):
    date_info = file_spec['date']
    date_fields = date_info['field']
    date_string = data
    for f in date_fields:
        date_string = date_string[f]
    date = dt.strptime(date_string, date_info['parse_format'])
    if not as_datetime_object:
        date = date.strftime(date_info['report_format'])
    return date


def distill_to_dict(file_path, file_spec, np_array=False):
    '''Takes xls csv dump and extracts data according to the given file specifications, returns a string'''
    aggregated = dict()
    for k, spec in file_spec.items():
        data = read_spec(contents, spec)
        spec_type = spec[0]
        if spec_type == 'table':
            labels = data[0]
            n = len(labels)
            aggregated[k+'_columns'] = labels 
            if np_array:
                table = np.array(data[1:], dtype=float).dropna()
            else:
                table = [r for r in data[1:] if len(r) == n]
            aggregated[k] = table
        elif spec_type == 'params':
            entry = dict()
            for r in data:
                entry[r[0]] = r[1]
            aggregated[k] = entry
    return aggregated



def read_xls(file_path, file_spec):
    '''
    Parses a xls document at file_path into a python dictionary. 
    Given a file_spec in dict format.
    '''
    global current_xls_file
    current_xls_file = file_path
    #contents = xls_dump(converter_path=converter, in_file=file_path, out_file=None)
    distillate = distill_to_dict(file_path, file_spec['file_spec'])
    # if 'date' in file_spec:
    #     date_spec = file_spec['date']
    #     v = distillate
    #     for k in date_spec['field']:
    #         v = v[k]
    #     datetime_obj = dt.strptime(v, date_spec['parse_format'])
    #     distillate['analysis_date'] = datetime_obj.strftime(date_spec['report_format'])
    #     distillate['timestamp_analysis'] = datetime_obj.timestamp()
    distillate['raw_file'] = file_path
    distillate['CYCLES_N'] = len(distillate['CYCLES'])
    distillate['timestamp_readxls'] = dt.now().timestamp()
    current_xls_file = ''
    return distillate