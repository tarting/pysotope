'''
pysotope - a package for inverting double spike isotope analysis data
This module contains the commandline tool for running the data reduction
procedure.
This module should under no circumstances be imported to a another module
in pysotope.
'''

# TODO: Refactor helper functions

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


import sys
import os
from glob import glob
import re
import json
import functools

import click
from tqdm import tqdm
import pandas as pd

import pysotope as pst

str_sum = lambda s: functools.reduce(lambda a, b: a+b, s)


def read_list(list_file):
    return pd.read_excel(list_file)

def locate_spec_file(working_dir=None):
    if working_dir is None:
        working_dir = os.getcwd()

    click.echo('STATUS | no spec-file supplied, looking in {}'.format(working_dir))
    candidates = glob(os.path.join(working_dir, '*.json'))
    click.echo('       | found {}'.format(candidates))
    if len(candidates) == 0:
        click.echo('       | no spec-file found please provide one')
        spec = {}
    else:
        click.echo('       | using {}'.format(candidates[0]))
        spec = pst.read_json(candidates[0])
    return spec


def collate_cycles(all_data, summaries_df):
    cycle_df = pd.DataFrame()
    click.echo('\rSTATUS | Collecting cycles ...', err=True)
    for k, v in tqdm(all_data.items()):
        run_df = pd.DataFrame(v)
        run_df['sample_id'] = summaries_df.loc[k, 'sample_id']
        run_df['sample_text'] = summaries_df.loc[k, 'sample_text']
        run_df['bead_id'] = summaries_df.loc[k, 'bead_id']
        run_df['run_no'] = summaries_df.loc[k, 'run_no']
        run_df['cycle'] = [i+1 for i in run_df.index]
        
        run_df.index = ['{} {} {}'.format(b, r, c) for b, r, c in zip(run_df['bead_id'], run_df['run_no'], run_df['cycle'])]
        
        cycle_df = cycle_df.append(run_df)
    return cycle_df


def trim_table(t_columns, row):
    new_table = {}
    first = row['first_row'] - 1
    end = row['last_row']
    for k, v in t_columns.items():
        n_rows = len(v)
        if n_rows < end:
            end = n_rows
        
        new_table[k] = v[first:end]
    return new_table

        

def reduce_data(overview_df, spec):
    # Need to add a fail case for missing spec file data
    overview_df = pst.filelist.verify_file_list(overview_df)
    all_data = {}
    all_summaries = {}
    click.echo('STATUS | Processing data ...', err=True)
    #no_read = []
    s_labels = []
    items = list(overview_df.iterrows())
    for k, row in tqdm(items):
        filepath = row['filepath']
        if row['ignore']:
            continue
        sys.stdout.flush()
        filename, data = pst.safe_read_file(filepath, spec)
        t_labels, t_columns = pst.safe_invert_data(data, spec)
        t_columns = trim_table(t_columns, row)
        s_labels_temp, s_row = pst.safe_summarise_data(t_labels, t_columns, spec)        
        # Make sure that the final labels list contains values.
        # Does not succeed if no data was reduced.
        if s_labels == []:
            s_labels = s_labels_temp
        
        if s_row != []:
            s_row += list(overview_df.loc[filename])
        if t_labels != []:
            all_data[filename] = t_columns
        
        if s_labels_temp != []:
            all_summaries[filename] = s_row 
    
    # Add filepath key to summary labels
    s_labels += list(overview_df.columns)
    summaries_df = pd.DataFrame(all_summaries, index=s_labels).T
    summaries_df['spl_conc'] = summaries_df.F_conc * summaries_df.spk_wt * summaries_df.spk_conc / summaries_df.spl_wt
    
    # Collect all cycles in a single dataframe
    cycles_df = collate_cycles(all_data, summaries_df)

    return summaries_df, cycles_df





@click.group()
@click.option('-v', '--verbose', count=True)
@click.pass_context
def main(ctx, verbose):
    if verbose:
        print('Árgh')
    else:
        pass

@main.command()
@click.argument('datadir')
@click.argument('listfile', required=False, default='./external_variables.xlsx')
@click.pass_obj
def init(ctx, datadir, listfile):
    new_list = pst.filelist.append_to_list(datadir, listfile)
    new_list.to_excel(listfile)

@main.command()
@click.argument('listfile')
@click.argument('specfile', required=False)
@click.argument('outfile', required=False)
@click.pass_obj
def invert(ctx, listfile, specfile, outfile):    
    click.echo('running invert command')

    if outfile is None:
        outfile = './results'

    if specfile is None:
        spec = locate_spec_file()
    elif os.path.isdir(specfile):
        spec = locate_spec_file(specfile)
    elif os.path.isfile(specfile):
        spec = pst.read_json(specfile)
    else:
        spec = {}

    if spec == {}:
        click.echo('ERROR  | Specification-file not found {}'.format(specfile), err=True, color='red')
    else:
        if os.path.isfile(listfile):
            overview_df = pd.read_excel(listfile)
            summaries, cycles = reduce_data(overview_df, spec)
            summaries.to_excel(outfile + '.xlsx')
            cycles.to_csv(outfile + '_cycles.csv')

        else:
            click.echo('ERROR  | Supplied list-file does not exist {}'.format(listfile), err=True, color='red')

if __name__ == '__main__':
    main()
