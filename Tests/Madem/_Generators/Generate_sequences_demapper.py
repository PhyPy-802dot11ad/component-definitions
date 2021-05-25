"""Generate random complex sequences.
"""

import os

import numpy as np


np.random.seed( 0 )

complex_sequences = \
    np.random.random( (100,120) ) +\
    1j * np.random.random( (100,120) )

binary_sequences = np.random.randint( 0, 2, (100,120), dtype='int8' )

basedir = os.path.abspath(os.path.dirname(__file__))
input_path = os.path.join( basedir, '..', '_Input' )

np.save( os.path.join( input_path, 'complex_sequences_demapper.npy' ), complex_sequences)
np.save( os.path.join( input_path, 'binary_sequences_demapper.npy' ), binary_sequences)
