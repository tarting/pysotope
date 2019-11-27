# pysotope

A Python 3 module and command line tool for reducing Double Spike Isotope
measurements acquired on the IsoprobeT and Phoenix instruments at the
Geology Section at the University of Copenhagen. 


# User guide

## Installing

Get a working install of python e.g. via the Anaconda python distribution
(Used for testing on Windows10, MacOS Mojave, and Manjaro GNU/Linux).

Install the package by downloading/cloning this repository navigating to
the cloned folder and running:

```sh
# To clone (requires git)
git clone https://github.com/tarting/pysotope
cd pysotope
python setup.py install --user
```

This will install the packages listed in the requirements.txt file as well
as the pysotope library and command.

Check that pysotope is installed by importing the pysotope library in
python: Run ```python [return]``` in the commandline to launch the python
shell followed by: ```import pysotope [return]```. If no errors are
reported, pysotope is installed correctly. Type ```exit() [return]``` to
exit the python shell.


## Using the command-line tools

These tools are specifically made for the instruments, naming-convention and
workflow at the UCPH labs.

Pysotope requires a specific directory structure, and works best if
a separate folder is used per sample-set or project. The data_root directory
must contain one appropriate `.json` specification file and a folder
containing the data as .xls files or .raw folders from the IsoWorks
software. 

```sh
| project_root/

    | Cd_data_root/
        | Cd_spec_file.spec.json
        | data_dir/
            | run1 2189.xls
            | run2 2190.xls
            ...
            
    | Cr_data_root/
        | Cr_spec_file.spec.json
        | data_dir/
            | run1.raw/
              | run1 2191.xls
            | run2.raw/
              | run2 2192.xls
            ...

```

Running pysotope is then a matter of opening a console (e.g. anaconda
prompt on windows, or a terminal on MacOS and Linux. 

Navigate to the data_root directory using the `cd` commmand and launch the
pysotope command in the following order.

```sh
pysotope init 'data_dir'
```

This command generates a list of datafiles saved as external_variables.xlsx. Now
modify this file to include sample weight, spike weight, and spike concentration.
The existing columns can be edited to exclude parts of the recorded data e.g. in
case of missing signal, to exclude entire runs without signal etc.
Any data in columns added to the right of the initial rows will be carried
on to the final results file. Any modifications made to the list persist
through reruns of the init command. In this way you can add more data file
by simply dropping them into the data directory an re-running the pysotope init
command. This command will prompt you for a .spec.json file if no file is in the
project directory.

```sh
pysotope invert 'external_variables.xlsx'
```

This command uses the list generated by the init command and the specification
file to invert the double spike data, and produces a results.xlsx with summarized
data for each run, and an results_cycles.csv containing each collected cycle for
every run.

```sh
pysotope plot results.xlsx
```

This command creates summary diagrams for each bead, and collected summary
for all runs. These are put as png files in a GFX folder. The summary
diagrams display each run in separate colors, cycles excluded from mean
calculation as red crosses, as well as 2 standard deviation and standard
error fields, for both individual runs and summarized across a bead run.

## Calibrating spike isotope composition

The spike isotope composition can be calibrated using the calibrate command given 
an uncalibrated spec file, and an external variables file containing standard runs
to calibrate against.

```sh
pysotope calibrate 'external_variables_SRM.xlsx' 'uncalibrated.spec.json' 'output.spec.json'
```

This command produces a new .spec.json file calibrated such that the beam-intensity
<<<<<<< HEAD
weighted average of the isotope composition of interrest is 0, e.g. δ⁵³Cr.
=======
weighted average of the isotope composition of interrest is 0, e.g. ∂^53^Cr.
>>>>>>> d3c6f89bedcaef7844fec6af42857723764a396d

## Using as a python module

Import pysotope:
```python
import pysotope as pst
```

You still need to provide a specification file (described in previous
section), to provide data external to the measurement.

```python
spec = pst.read_json('pysotope/spec/Cr-reduction-scheme-data_only.json')
```

Read the data such that each cycle represented as a list, numpy.array, or
pd.Series of float values in a dictionary containing the key 'CYCLES'.
Given an excel sheet, csv or pandas dataframe of the format:

| Index | m49 | m50 | m51 | m52 | m53 | m54 | m56 |
| ----: | --: | --: | --: | --: | --: | --: | --: |
|     0 | 0.0 | 0.2 | 0.0 | 0.5 | 0.3 | 0.2 | 0.0 |
|     1 | 0.0 | 0.3 | 0.0 | 0.6 | 0.4 | 0.3 | 0.0 |
|     2 | 0.0 | 0.2 | 0.0 | 0.5 | 0.3 | 0.2 | 0.0 |
|   ... | ... | ... | ... | ... | ... | ... | ... |
|   120 | 0.0 | 0.2 | 0.0 | 0.5 | 0.3 | 0.2 | 0.0 |

Where the index of the data columns is specified in the spec-file, in
this case with ```pd.read_[filetype](<file>, index_col=0)``` index of m49
is 0 and m56 is 6. There can be any number of columns in the datafile as
long as it is indexed correctly.

```python
import pandas as pd

#               sample-ID note  Date-ISO   no-serial number
df = pd.read(
    'datadir/subdir/SPLID-015 500mV 2019-09-01 01-9999.xls',
    index_col=0)
```

Invert the data:

```python
reduced_cycles = pst.invert_data(df.values, spec)
```

Where reduced_cycles is an OrderedDict with string index and 
numpy.ndarray[float] values.


And summarise:
```python
summary_statistics = pst.summarise_data(reduced_cycles, spec)
```

Which is an OrderedDict with string index and float values. These can e.g.
be joined into a pandas dataframe, or be written to a csv file.




