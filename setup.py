#!/usr/bin/env python3

from setuptools import setup, find_packages

requirements_handle = open('requirements.txt','r')
requirements = requirements_handle.read().splitlines()
requirements_handle.close()

setup(
    name='pysotope',
    version='0.1',
    description='Invert double spike isotope data.',
    author='Trygvi Bech Arting',
    author_email='trygvi@gmail.com',
    url='https://github.com/tarting/pysotope',
    license='GPL-2',
    packages=find_packages(),
    data_files=[('pysotope', 
    	['pysotope/spec/Cr-reduction_scheme.json',
         'pysotope/spec/Cd-reduction_scheme.json',
         'bin/xls2csv'])],
    install_requires=requirements,
#    entry_points={
#        'console_scripts': ['pynvert = pysotope.invert:main']
#    },
)
