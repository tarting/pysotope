'''
pysotope - a package for inverting double spike isotope analysis data
Data reading module provides a very basic plugin architecture for filetypes.
Python files exposing a read file function can
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

import sys
import json
import importlib

import pysotope.ratios as ratios
from pysotope.typedefs import Spec, Data, Any, List, Dict, Union


# Reading specification files
def read_json(
        file_path: str,
        ) -> Spec:
    '''
    Read json document to Spec
    '''
    with open(file_path, 'r') as file_handle:
        file_spec = json.load(file_handle)
    return file_spec


def read_spec_file(
        file_path: str,
        ) -> Spec:
    '''
    Read reduction specification file
    '''
    spec = read_json(file_path)
    if 'version' in spec:
        version = spec['version']
    else:
        version = 1

    if version == 1:
        spike = ratios.calc_spec_abund(
                'spike', 'report_fracs', spec)
        spec['spike'] = spike

        standard = ratios.calc_spec_abund(
                'standard', 'report_fracs', spec)
        spec['standard'] = standard

    return spec

def check_plugin_exists(module_name):
    if sys.version_info < (3, 4):
        import pkgutil
        loader = pkgutil.find_loader(module_name)
    elif sys.version_info >= (3, 4):
        loader = importlib.util.find_spec(module_name)

    return loader is not None

# Reading data files
class DataReader(object):

    def __init__(self, spec):
        self.filetype_plugin = spec['filetype_plugin']
        self.load_spec(spec)

    def load_spec(self, spec):
        self.spec = spec

        self.load_plugin(self.filetype_plugin)
        self.filetype_plugin = self.spec['filetype_plugin']

    def load_plugin(self, plugin_name):
        if check_plugin_exists(plugin_name):
            self.filetype_module = importlib.import_module(
                name='{}'.format(plugin_name),
            )

        else:
            print('Error: Module {} not found'.format(plugin_name))

    def read_file(self, filepath):
        self.data = self.filetype_module.read_file(filepath, self.spec)
        return self.data

    def __call__(self, filepath):
        return self.read_file(filepath)


# For backwards compatibility
def read_xls(file_path, spec):
    '''
    Deprecated
    Can in principle read any filetype supported by available plugins
    '''

    print("Deprecation waringn: pst.data_reader.read_xls is deprecated",
          file=sys.stderr)

    if "filetype_plugin" not in spec:
        spec['filetype_plugin'] = 'pysotope.plugins.filetype_xls'

    file_reader = DataReader(spec)
    data = file_reader(file_path)

    return data
