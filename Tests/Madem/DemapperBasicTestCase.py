"""
Test input/output data type and dimensions.

"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents.Madem import Mapper, Demapper


class DemapperBasicTestCase(unittest.TestCase):

    def setUp(self):

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_complex_sequences = np.load( os.path.join( basedir, '_Input', 'complex_sequences.npy') )

        self.Rm = [1, 2, 4, 6]

        self.noise_variance = 0.1

        self.demapper = Demapper()


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_sequence_input_type(self):
        """Verify passing the wrong input data types fails.
        """

        self.demapper.set_modulation_rate(self.Rm[2])

        sequences = [
            1,
            1.2,
            '1010',
            np.zeros((1, 9), dtype=float)
        ]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                with self.assertRaises(TypeError):
                    self.demapper.run( sequence, self.noise_variance )


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_single_dimension_input(self):
        """Verify passing a single dimension array fails.
        """

        for Rm in self.Rm:
            with self.subTest(modulation_rate=Rm):
                self.demapper.set_modulation_rate(Rm)

                sequence = np.zeros(Rm*9, dtype=complex)
                with self.assertRaises(ValueError):
                    self.demapper.run( sequence, self.noise_variance )


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_output_type(self):
        """Verify the output type is int.
        """

        complex_sequences = self.input_complex_sequences[0,None,:]

        for complex_sequence_idx, complex_sequence in enumerate(complex_sequences):
            with self.subTest(complex_sequence_idx=complex_sequence_idx):
                for Rm in self.Rm:
                    with self.subTest(modulation_rate=Rm):
                        self.demapper.set_modulation_rate(Rm)

                        demapper_sequence = self.demapper.run( complex_sequence[None,:], self.noise_variance )

                        self.assertTrue(
                            issubclass(demapper_sequence.dtype.type, np.float)
                        )


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_output_dimensions(self):
        """Verify the output dimensions are correct.
        """

        complex_sequences = self.input_complex_sequences[0,None,:]

        for complex_sequence_idx, complex_sequence in enumerate(complex_sequences):
            with self.subTest(complex_sequence_idx=complex_sequence_idx):
                for Rm in self.Rm:
                    with self.subTest(modulation_rate=Rm):
                        self.demapper.set_modulation_rate(Rm)

                        demapper_sequence = self.demapper.run( complex_sequence[None,:], self.noise_variance )

                        self.assertEqual(
                            demapper_sequence.size, complex_sequence.size * Rm
                        )
