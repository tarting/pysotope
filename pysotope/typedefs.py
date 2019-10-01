'''
Type definitions for pysotope
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


from typing import Dict, Any, Tuple, List, Callable
import numpy as np

Spec = Dict[str, Any]
Data = Dict[str, Any]

RowMx = np.ndarray
Row = np.ndarray
IntensRow = np.ndarray
RatioRow = np.ndarray
Column = np.ndarray
AlphaBetaLambda = np.ndarray
