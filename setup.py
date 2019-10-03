#!/usr/bin/env python3

from setuptools import setup, find_packages

requirements_handle = open('requirements.txt','r')
requirements = requirements_handle.read().splitlines()
requirements_handle.close()

setup(
    name='pysotope',
    version='0.2.1-testing',
    description='Invert double spike isotope data.',
    author='Trygvi Bech Árting',
    author_email='trygvi@gmail.com',
    url='https://github.com/tarting/pysotope',
    license='GPL-2',
    include_package_data=True,
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': ['pysotope = pysotope.run:main']
    },
)
