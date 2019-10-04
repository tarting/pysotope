#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as req_handle:
    requirements = req_handle.read().splitlines()

with open('README.md', 'r') as readme_handle:
    long_description = readme_handle.read()

setup(
    name='pysotope',
    version='0.2.1-testing',
    description='Invert double spike isotope data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    python_requires='>=3.6',
)
