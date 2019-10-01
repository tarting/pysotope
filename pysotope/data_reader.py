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
from functools import reduce
from datetime import datetime as dt
from dateutil import parser as dtparser

import numpy as np

from pysotope.exceptions import UndefinedDataDefinition, UnknownPlatform
from pysotope.typedefs import Spec, Data, Any, List


# Move to settings module, only used in the read_xls function
# path to a converter which dumps the contents of an xls to stdout
# as csv, appending all worksheets.
# Currently using binary shipped with the Haskell xls module.
if sys.platform == 'win32':
    CONVERTER = 'xls2csv.exe'
elif sys.platform == 'darwin':
    CONVERTER = 'xls2csv_macos_x86-64'
elif sys.platform == 'linux':
    CONVERTER = 'xls2csv'
else:
    raise UnknownPlatform('Platform [{}] unknown'.format(sys.platform))


CONVERTER_PATH = os.path.abspath(os.path.join(
        os.path.split(__file__)[0],
        'bin',
        CONVERTER))


def read_json(
        file_path: str,
        ) -> Spec:
    '''
    Read json document to Spec
    '''
    with open(file_path, 'r') as file_handle:
        file_spec = json.load(file_handle)
    return file_spec


def xls_dump(
        conv_path: str,
        in_file: str,
        out_file: str = None
        ) -> str:
    '''
    Dumps the content of an xls document into a contiguous csv string.
    Optionally dumps the string into a UTF-8 formatted file.
    '''
    if os.path.isfile(in_file):
        in_path = os.path.abspath(in_file)
        output = subprocess.check_output(
            [conv_path, in_path], universal_newlines=True)
        if out_file is not None:
            with open(out_file, 'w') as file_handle:
                file_handle.write(output)
    else:
        raise FileNotFoundError('Input file not found', in_file)

    return output


def get_table(
        contents: str,
        start_string: str,
        end_string: str,
        first_col: int,
        n_columns: int,
        skip_rows: int = 1,
        ) -> List[List[float]]:
    '''
    Read a data table by reading keywords, gets set number of columns.
    Skips row with keyword by default.
    '''
    start = contents[:contents.find(start_string)].rfind('\n') + 1
    table = contents[start:]
    table = table[:table.find(end_string)]
    table = table.splitlines()
    table = [l.split(',') for l in table]
    table = [l for l in table if len(reduce(lambda a, b: a+b, l)) > 0]
    table = [l[first_col:first_col+n_columns] for l in table[skip_rows:]]
    table = [[float(v) for v in r] for r in table]
    return table


def get_keyword_row(
        contents: str,
        search_string: str,
        first_col: int,
        n_columns: int,
        ) -> List[str]:
    '''
    Read a row containing a keyword. For fetching headers.
    '''
    start = contents[:contents.find(search_string)].rfind('\n') + 1
    line = contents[start:]
    line = line[:line.find('\n')]
    line = line.split(',')
    return line[first_col:first_col + n_columns]


def get_params(
        contents: str,
        labels: List[str],
        start_string: str,
        ) -> List[str]:
    '''
    Read value following comma after keyword.
    '''
    ctrl = contents[contents.find(start_string):]
    results = []
    for label in labels:
        res = ctrl[ctrl.find(label):]
        res = res[:res.find('\n')].split(',')[1]
        results.append(res)
    return results


def read_spec(
        contents: str,
        spec: Spec,
        ) -> Data:
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


def distill_to_csv(
        contents: str,
        file_spec: Spec
        ) -> str:
    '''Takes xls csv dump and extracts data according to the given file specifications, returns a string'''
    aggregated = str()
    for k, spec in file_spec.items():
        aggregated += k + '\n'
        data = read_spec(contents, spec)
        for r in data:
            aggregated += ','.join(r) + '\n'
    return aggregated


def distill_to_dict(
        contents: str,
        file_spec: Spec,
        np_array: bool = False
        ) -> Data:
    '''Takes xls csv dump and extracts data according to the given file specifications, returns a string'''
    aggregated = dict()
    for k, spec in file_spec.items():
        data = read_spec(contents, spec)
        spec_type = spec[0]
        if spec_type == 'table':
            aggregated[k+'_columns'] = data[0]
            aggregated[k] = np.array(data[1:], dtype=float) if np_array else data[1:]
        elif spec_type == 'params':
            entry = dict()
            for r in data:
                entry[r[0]] = r[1]
            aggregated[k] = entry
    return aggregated

def parse_date(
        data: Data,
        spec: Spec,
        ) -> dt:
    keys = spec['date']['field']
    analysis_start_time = data[keys[0]][keys[1]]
    dtobj = dtparser.parse(analysis_start_time)
    return dtobj

def read_xls(
        file_path: str,
        file_spec: Spec,
        ) -> Data:
    '''
    Parses a xls document at file_path into a python dictionary.
    Given a file_spec in dict format.
    '''
    contents = xls_dump(
        conv_path=CONVERTER_PATH, in_file=file_path, out_file=None)
    distillate = distill_to_dict(contents, file_spec['file_spec'])

    if 'date' in file_spec:
        try:
            fmt = file_spec['date']['report_format']
            dtobj = parse_date(distillate, file_spec)
            distillate['analysis_time'] = dtobj.strftime(fmt)
            distillate['analysis_timestamp'] = dtobj.timestamp()
        except KeyError as error:
            print(error)

    distillate['raw_file'] = file_path
    distillate['CYCLES_N'] = len(distillate['CYCLES'])
    distillate['timestamp_readxls'] = dt.now().timestamp()
    return distillate
