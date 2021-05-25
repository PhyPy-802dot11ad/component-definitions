"""Generate random sequences.
"""

import os

import numpy as np


N = 100

np.random.seed( 0 )

input_sequences = np.ones((N, 1024), complex)


basedir = os.path.abspath(os.path.dirname(__file__))

input_path = os.path.join( basedir, '..', '_Input' )

np.save( os.path.join( input_path, 'sequences.npy' ), input_sequences)
