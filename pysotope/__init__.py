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


from datetime import datetime as dt
from collections import namedtuple
import os
import sys

import click

from pysotope.data_reader import read_xls, read_json, parse_date
from pysotope.invert import invert_data, summarise_data, exp_corr
import pysotope.filelist as filelist
from pysotope.diagrams import generate_cycleplots, generate_summaryplot
from pysotope.run import reduce_data

labeled = namedtuple('labeled', ['labels', 'data'])
reduced = namedtuple('reduced', ['summary', 'data'])


def get_xls_inverter_from_spec(file_spec):
    def invert_xls(file_path):
        data = read_xls(file_path, file_spec)
        labels, results = invert_data(data['CYCLES'], file_spec)

        summary_labels, summary_results = summarise_data(labels, results, file_spec)
        
        summary_labels = ['file_path'] + summary_labels
        summary_results = [file_path] + summary_results
        
        summary_labels.append('file_spec_path')
        if 'file_spec_path' in file_spec:
            summary_results.append(file_spec['file_spec_path'])
        else: 
            summary_results.append('N/A')

        summary_labels.append('proc_time')
        summary_results.append(dt.now().strftime(file_spec['date']['report_format']))

        results = [*zip(*[list(results[l]) for l in labels])]\
    
        data['REDUCED'] = results
        data['REDUCED_columns'] = labels

        return reduced(labeled(summary_labels, summary_results), data)
    return invert_xls

def get_xls_inverter(file_spec_path):
    file_spec = read_json(file_spec_path)
    file_spec['file_spec_path'] = file_spec_path
    return get_xls_inverter_from_spec(file_spec)

def invert_from_paths(xls_path, file_spec_path):
    invert_xls = get_xls_inverter(file_spec_path)
    reduced = invert_xls(xls_path)
    return reduced


def safe_read_file(filepath, spec):
    filename = os.path.split(filepath)[1][:-4]
    try: 
        data = read_xls(filepath, spec)
    except Exception as e:
        sys.stdout.flush()
        click.echo('\rERROR  | while reading file {}: {}'.format(
					click.format_filename(filepath), e), 
              err=True)
        data = {}
    return filename, data

def safe_invert_data(data, spec):
    if data == {}:
        labels = []
        rows = []
    else:
        try:
            labels, rows = invert_data(data['CYCLES'], spec)
        except Exception as e:
            labels = []
            rows = {}
            sys.stdout.flush()
            click.echo('\rERROR  | while reducing file: {}'.format(e), 
                  err=True)
    return labels, rows

def safe_summarise_data(t_labels, t_rows, spec):
    try: 
        s_labels, s_row = summarise_data(t_labels, t_rows, spec)
    except Exception as e:
        s_labels = []
        s_row = []
        sys.stdout.flush()
        click.echo('\rERROR  | while summarising file: {}'.format(e), 

              err=True)
    return s_labels, s_row
