"""Scrambler class test case.
"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents.Scrambler import Scrambler


class ScramblerTestCase(unittest.TestCase):


    def setUp(self):
        """Fetch test input data.
        """
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_registers = np.load( os.path.join( basedir, '_Input/register.npy') )
        self.input_sequences = np.load( os.path.join( basedir, '_Input/sequence.npy') )
        
        self.scrambler = Scrambler()

    @unittest.skip('Data checking omitted in research package implementation.')
    def test_register_input_type(self):
        """Verify the scrambler raises a TypeError for incorrect register input data types.
        """
        
        registers = [
            1,
            1.2,
            '1010',
            np.zeros((1,7), dtype=complex)
        ]
        
        for register_idx, register in enumerate(registers):
            with self.subTest( register_idx=register_idx ):
                with self.assertRaises(TypeError):
                    self.scrambler.set_register(register)

    @unittest.skip('Data checking omitted in research package implementation.')
    def test_register_input_dimensions(self):
        """Verify the scrambler raises a ValueError for incorrect register input dimensions.
        """
        
        registers = [
            np.zeros(7, dtype='int8'),
            np.zeros((1,8), dtype='int8'),
            np.zeros((1,6), dtype='int8'),
            np.zeros((2,7), dtype='int8')
        ]
        
        for register_idx, register in enumerate(registers):
            with self.subTest( register_idx=register_idx ):
                with self.assertRaises(ValueError):
                    self.scrambler.set_register(register)


    def test_register_setter_and_getter(self):
        """Verify new register values get loaded correctly.
        """

        for register_idx, register in enumerate(self.input_registers):
            with self.subTest( register_idx=register_idx ):
                self.scrambler.set_register(register)
                np.testing.assert_equal( self.scrambler.get_register(), register )

    @unittest.skip('Data checking omitted in research package implementation.')
    def test_sequence_input_type(self):
        """Verify the scrambler raises a TypeError for incorrect sequence input data types.
        """

        sequences = [
            1,
            1.2,
            '1010',
            np.zeros((1,1024), dtype=complex)
        ]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest( sequence_idx=sequence_idx ):
                with self.assertRaises(TypeError):
                    self.scrambler.run(sequence)

    @unittest.skip('Data checking omitted in research package implementation.')
    def test_sequence_input_dimensions(self):
        """Verify the scrambler raises a ValueError for incorrect sequence input dimensions.
        """

        sequences = [
            np.zeros(1024, dtype='int8'),
            np.zeros((2,1024), dtype='int8')
        ]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest( sequence_idx=sequence_idx ):
                with self.assertRaises(ValueError):
                    self.scrambler.run(sequence)


    def test_sequence_scrambling(self):
        """Verify scrambling and descrambling a sequence returns the initial input data.
        """
        
        for sequence_idx, (register, sequence) in enumerate(zip(self.input_registers, self.input_sequences)):
            with self.subTest( sequence_idx=sequence_idx ):

                self.scrambler.set_register(register)
                scrambled = self.scrambler.scramble(sequence)

                self.scrambler.set_register(register)
                descrambled = self.scrambler.scramble(scrambled)

                np.testing.assert_equal( descrambled, sequence )
