'''
        iry:
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
import re
import shutil
import functools
from glob import glob
from collections import OrderedDict

import click
import pandas as pd
from tqdm import tqdm

import pysotope as pst
from pysotope.typedefs import Spec, List, Tuple


def str_sum(
        strings: List[str]
        ) -> str:
    '''Join a list of strings'''
    joined = functools.reduce(lambda a, b: a + b, strings)
    return joined


def read_list(
        list_file
        ) -> pd.DataFrame:
    return pd.read_excel(list_file)


def locate_spec_file(
        working_dir: str = None
        ) -> Spec:
    '''Identify and read spec file'''
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
        spec = pst.read_spec_file(candidates[0])
    return spec


def collate_cycles(
        all_data: pd.DataFrame,
        summaries_df: pd.DataFrame
        ) -> pd.DataFrame:
    '''
    Append run and sample data to cycles
    Obsolete as of v0.2.2
    '''
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


# write_cycles(cycles_file, reduced, summary)
def write_cycles(
        cycles_file: str,
        reduced: pd.DataFrame,
        summary: pd.Series,
        ) -> None:
    '''
    Append cycles to file.
    '''
    cycles = pd.DataFrame(reduced)

    cycles['sample_id'] = summary['sample_id']
    cycles['sample_text'] = summary['sample_text']
    cycles['bead_id'] = summary['bead_id']
    cycles['run_no'] = summary['run_no']
    cycles['cycle'] = [i+1 for i in cycles.index]

    cycles['analysis_start_time'] = summary['analysis_time']
    cycles['analysis_start_timestamp'] = summary['analysis_timestamp']

    cycles.index = [
        '{} {:2.0f} {}'.format(b, r, c)
        for b, r, c in zip(
            cycles['bead_id'],
            cycles['run_no'],
            cycles['cycle'])
        ]
    if os.path.isfile(cycles_file):
        with open(cycles_file, 'a') as file_handle:
            cycles.to_csv(file_handle, header=False)
    else:
        cycles.to_csv(cycles_file)

    return cycles

def trim_table(
        t_columns,
        row,
        ):
    new_table = OrderedDict()
    first = row['first_row'] - 1
    end = row['last_row']
    for k, v in t_columns.items():
        n_rows = len(v)
        if n_rows < end:
            end = n_rows

        new_table[k] = v[first:end]
    return new_table

def reduce_data(
        overview_df: pd.DataFrame,
        spec: Spec,
        cycles_file: str = None,
        ) -> pd.DataFrame:
    # Need to add a fail case for missing spec file data
    if os.path.isfile(cycles_file):
        os.remove(cycles_file)

    overview_df = pst.filelist.verify_file_list(overview_df)
    all_summaries = OrderedDict()
    click.echo('STATUS | Processing data ...', err=True)
    # no_read = []
    items = list(overview_df.iterrows())
    for k, row in tqdm(items):
        filepath = row['filepath']
        if row['ignore']:
            continue
        sys.stdout.flush()
        filename, data = pst.safe_read_file(filepath, spec)
        reduced = pst.invert_data(data['CYCLES'], spec)
        reduced = trim_table(reduced, row)
        summary = pst.summarise_data(reduced, spec)
        try:
            time = data['analysis_time']
            timestamp = data['analysis_timestamp']
        except KeyError:
            time = ''
            timestamp = 0.
        summary['analysis_time'] = time
        summary['analysis_timestamp'] = timestamp

        # Make sure that the final labels list contains values.
        # Does not succeed if no data was reduced.

        if summary:
            summary.update(overview_df.loc[filename])
            all_summaries[filename] = summary
        if bool(reduced) & (cycles_file is not None):
            write_cycles(cycles_file, reduced, summary)
            

    # Add filepath key to summary labels
    s_labels = summary.keys()
    summaries_df = pd.DataFrame(all_summaries, index=s_labels).T
    summaries_df['spl_conc'] = (
        summaries_df.F_conc *
        summaries_df.spk_wt *
        summaries_df.spk_conc /
        summaries_df.spl_wt)

    return summaries_df





@click.group()
@click.option('-v', '--verbose', count=True)
@click.pass_context
def main(
        ctx: dict,
        verbose: bool
        ) -> None:
    if verbose:
        pass
    else:
        pass

def choose_file(files):
    for i, file_path in enumerate(files):
        file_name = os.path.split(file_path)[1]
        click.echo('{:>3d}: {}'.format(i+1, file_name))
    choice = click.prompt(
        'Choose spec file, {}-{}'.format(1, len(files)),
        default=0, type=int) - 1

    try:
        file_path = files[choice]
    except IndexError:
        file_path = ''

    return file_path
    

def get_spec_from_store(spec_file):
    old_sf = locate_spec_file()
    
    scr_path = os.path.split(__file__)[0]
    spec_path = os.path.join(scr_path, 'spec')
    click.echo(spec_path)
    spec_files = sorted(glob(os.path.join(spec_path, '*.json')))
    matches = []
    if spec_file:
        for file_path in spec_files:
            m = re.findall(
                    spec_file.lower(),
                    file_path.lower())
            if m:
                matches.append(file_path)
    if not matches:
        matches = spec_files

    if len(matches) == 1:
        spec_file = matches[0]
    if len(matches) >= 2:
        spec_file = choose_file(matches)
    else:
        click.echo('No spec files were found in spec file folder:')
        click.echo('{}'.format(spec_path), )
        spec_file = ''

    return spec_file


@main.command()
@click.argument('spec_file', required=False, default='')
@click.pass_obj
def spec(
        ctx: dict,
        spec_file: str,
        ) -> None:
    
    spec_file = get_spec_from_store(spec_file)
    if spec_file: 
        shutil.copy2(spec_file, '.')

@main.command()
@click.argument('datadir')
@click.argument('listfile', required=False, default='./external_variables.xlsx')
@click.pass_obj
def init(
        ctx: dict,
        datadir: str,
        listfile: str,
        ) -> None:
    spec = locate_spec_file()
    if not spec:
        spec_file = get_spec_from_store('')
        if spec_file:
            shutil.copy2(spec_file, '.')
    new_list = pst.filelist.append_to_list(datadir, listfile)
    new_list.to_excel(listfile)

@main.command()
@click.argument('resultfile')
@click.argument('specfile', required=False)
@click.argument('gfxdir', required=False, default='./GFX')
@click.pass_obj
def plot(
        ctx: dict,
        resultfile: str,
        specfile: str,
        gfxdir: str
        ) -> None:
    if not os.path.isdir(gfxdir):
        os.mkdir(gfxdir)

    if specfile is None:
        spec = locate_spec_file()
    elif os.path.isdir(specfile):
        spec = locate_spec_file(specfile)
    elif os.path.isfile(specfile):
        spec = pst.read_spec_file(specfile)
    else:
        spec = {}

    if spec:
        if os.path.isfile(resultfile):
            idx = resultfile.rfind('.')
            result_base = resultfile[:idx]
        else:
            result_base = resultfile
        summary_file = result_base + '.xlsx'
        cycles_file = result_base + '_cycles.csv'

        summary_df = pd.read_excel(summary_file, index_col=0)
        cycle_df = pd.read_csv(cycles_file, index_col=0)

        pst.generate_summaryplot(summary_df, spec, gfxdir)
        pst.generate_cycleplots(cycle_df, summary_df, spec, gfxdir)
    else:
        click.echo('ERROR  | Specification-file not found {}'.format(specfile), err=True, color='red')


@main.command()
@click.argument('listfile')
@click.argument('specfile', required=False)
@click.argument('outfile', required=False)
@click.pass_obj
def invert(
        ctx: dict,
        listfile: str,
        specfile: str,
        outfile
        ) -> None:
    click.echo('running invert command')

    if outfile is None:
        outfile = './results'

    if specfile is None:
        spec = locate_spec_file()
    elif os.path.isdir(specfile):
        spec = locate_spec_file(specfile)
    elif os.path.isfile(specfile):
        spec = pst.read_spec_file(specfile)
    else:
        spec = {}

    if spec:
        if os.path.isfile(listfile):
            overview_df = pd.read_excel(listfile, index_col=0)
            summaries = reduce_data(
                overview_df,
                spec,
                cycles_file=outfile + '_cycles.csv')
            summaries.to_excel(outfile + '.xlsx')

        else:
            click.echo('ERROR  | Supplied list-file does not exist {}'.format(listfile), err=True, color='red')
    else:
        click.echo('ERROR  | Specification-file not found {}'.format(specfile), err=True, color='red')


if __name__ == '__main__':
    main()
