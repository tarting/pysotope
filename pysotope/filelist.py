'''
pysotope - a package for inverting double spike isotope analysis data
This module is for generating a list of datafiles in order to pass on information about spike
composition and weight, sample weight, and other run specific information not available in the
data files. The append_to_list function either appends newly discovered datafiles to an existing
list, or generates a new list based on all discovered files.
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
import re
from glob import glob

import pandas as pd
import click

def check_search_pattern(folder_pattern, extension='.xls'):
    click.echo('STATUS | Searching: {}'.format(folder_pattern), err=True, nl=False)
    run_files_all = glob(folder_pattern)
    run_files_all = [f for f in run_files_all if extension == f[-len(extension):]]
    if len(run_files_all) == 0:
        click.echo(' | No data-files found'.format(folder_pattern),err=True, color='red')
    else:
        click.echo(' | Found {} files'.format(len(run_files_all)), err=True, color='green')
    return run_files_all

def extract_pattern(pattern):
    def extract_func(f):
        found = re.findall(pattern, f)
        if len(found) > 0:
            found = found[0]
        else:
            found = ''
        return found
    return extract_func

def find_raw_files(folder_pattern, extension='.xls'):
    '''
    Extract info from filenames.
    '''

    # Find all data files
    run_files_all = check_search_pattern(folder_pattern, extension)
    if run_files_all == []:
        run_files_all = check_search_pattern(
                os.path.join(folder_pattern, '*/*{}'.format(extension)),
                extension,
        )
        if run_files_all == []:
            run_files_all = check_search_pattern(
                    os.path.join(folder_pattern, '*{}'.format(extension)),
                    extension)
            if run_files_all == []:
                click.echo('Error  | No data-files found.', err=True, color='red', bold=True)

    # Store filename as index in a Pandas DataFrame
    run_data = {os.path.split(p)[1][:-4]: [p] for p in sorted(run_files_all)}
    run_data = pd.DataFrame.from_dict(run_data, orient='index')
    run_data.columns =['filepath']

    # Add columns for sample and run identification
    run_data['filename'] = run_data.index
    run_data['sample_id'] = run_data.filename.apply(
            lambda f: os.path.split(os.path.split(f)[0])[1])

    run_data['sample_text'] = run_data.filename.apply(
            extract_pattern('(^[\s\S]+)\ [0-9]+-[0-9][0-9]-[0-9]+'))
    run_data['bead_id'] = run_data.filename.apply(
            extract_pattern('(^[\s\S]+\ [0-9]+-[0-9][0-9]-[0-9]+)'))
    run_data['run_no'] = run_data.filename.apply(
            extract_pattern('([0-9][0-9])-[0-9]+$'))


    # Extract date information
    dates = pd.Series()
    for i in run_data.index:
        year_first = re.findall('([0-9]{4})-([0-9]{2})-([0-9]{2})', i)
        year_last = re.findall('([0-9]{2})-([0-9]{2})-([0-9]{4})', i)
        if year_first != []:
            date = year_first[-1]
            dates[i] = '{}-{}-{}'.format(*date)
        elif year_last != []:
            date = year_last[-1]
            dates[i] = '{2}-{1}-{0}'.format(*date)
        else:
            dates[i] = ''
    run_data['date'] = dates


    # Add default data for spike and run
    run_data['first_row'] = 1
    run_data['last_row'] = 120
    run_data['ignore'] = False
    run_data['remarks'] = ''
    run_data['spl_wt'] = 1
    run_data['spk_wt'] = 1
    run_data['spk_conc'] = 1

    return run_data

def append_to_list(data_folder, list_file, extension='.xls'):
    '''
    Check for existing list_file and append new data if possible.
    '''
    run_data = find_raw_files(data_folder, extension)

    if os.path.isfile(list_file):
        file_df = pd.read_excel(list_file, index_col=0)
        n_rows_to_add = len(file_df) - len(run_data)
        click.echo('STATUS | Adding {} files to existing list_file {}'.format(n_rows_to_add, list_file),
                err=True)
        for i, row in run_data.iterrows():
            if i not in file_df.index:
                file_df.loc[i] = row
        final_list = file_df
    else:
        click.echo('STATUS | File {} not found, generating new list_file.', err=True)
        final_list = run_data

    return final_list

def verify_file_list(file_df):
    '''
    Check if data-files listed exist.
    Returns a dataframe without nonexistent files.
    '''
    for i, (idx, row) in enumerate(file_df.iterrows()):
        exists = os.path.isfile(row['filepath'])
        if exists:
            pass
        else:
            click.echo('Error  | File not found at row {}: {}'.format(
                i+2, row['filepath']), err=True, color='yellow')
            file_df = file_df.drop(idx)

    return file_df
