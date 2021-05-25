"""Generate random dataword sequences.
"""

import os

import numpy as np


np.random.seed( 0 )

# LLR_sequences = np.random.uniform( low=-100, high=100, size=(100,672))
sequences = np.random.randint( 0, 2, (100,546), dtype='int8' )

basedir = os.path.abspath(os.path.dirname(__file__))
input_path = os.path.join( basedir, '..', '_Input' )

# np.save( os.path.join( input_path, 'LLR_sequences.npy' ), LLR_sequences)
np.save( os.path.join( input_path, 'sequences_decoder.npy' ), sequences)
