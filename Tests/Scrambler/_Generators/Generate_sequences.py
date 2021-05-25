"""Generate random sequences.
"""

import os

import numpy as np


N = 100

np.random.seed( 0 )

input_registers = np.zeros((N, 7), dtype='int8')
for i in range(N):
    input_register = np.random.randint(0, 2, 7, dtype='int8')  # Generate random sequence
    # At least one element must be non-zero
    if input_register.max() == 0:
        input_register[0, np.random.randint(0, 7)] = 1  # Set element at random index to 1
    input_registers[i,:] = input_register

input_sequences = np.random.randint(0, 2, (N, 1024), dtype='int8')

basedir = os.path.abspath(os.path.dirname(__file__))

input_path = os.path.join( basedir, '..', '_Input' )

np.save( os.path.join( input_path, 'register.npy' ), input_registers)
np.save( os.path.join( input_path, 'sequence.npy' ), input_sequences)
