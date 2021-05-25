"""
Test input/output data type and dimensions.

"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents.Madem import Mapper


class MapperBasicTestCase(unittest.TestCase):

    def setUp(self):

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_sequences = np.load( os.path.join( basedir, '_Input', 'sequences_mapper.npy') )

        self.Rm = [1, 2, 4, 6]

        self.mapper = Mapper()


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_sequence_input_type(self):
        """Verify passing the wrong input data types fails.
        """

        self.mapper.set_modulation_rate(self.Rm[2])

        sequences = [
            1,
            1.2,
            '1010',
            np.zeros((1, self.Rm[2]*9), dtype=complex)
        ]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                with self.assertRaises(TypeError):
                    self.mapper.run(sequence)


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_single_dimension_input(self):
        """Verify passing a single dimension array fails.
        """

        for Rm in self.Rm:
            with self.subTest(modulation_rate=Rm):
                self.mapper.set_modulation_rate(Rm)

                sequence = np.zeros(Rm*9, dtype='int8')
                with self.assertRaises(ValueError):
                    self.mapper.run(sequence)


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_wrong_dimensions_input(self):
        """Verify passing a sequence of wrong dimensions fails.
        """

        for Rm in self.Rm:
            if Rm == 1: continue # SKip modulation rate 1
            with self.subTest(modulation_rate=Rm):
                self.mapper.set_modulation_rate(Rm)

                sequence = np.zeros((1, Rm*9-1), dtype='int8')
                with self.assertRaises(ValueError):
                    self.mapper.run(sequence)

                sequence = np.zeros((1, Rm*9+1), dtype='int8')
                with self.assertRaises(ValueError):
                    self.mapper.run(sequence)


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_output_type(self):
        """Verify the ouptu type is complex.
        """

        sequences = self.input_sequences[0,None,:]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                for Rm in self.Rm:
                    with self.subTest(modulation_rate=Rm):
                        self.mapper.set_modulation_rate(Rm)

                        mapped_sequence = self.mapper.run(sequence[None,:])

                        self.assertTrue(
                            issubclass(mapped_sequence.dtype.type, np.complex)
                        )


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_output_dimensions(self):
        """Verify the ouput dimensions are correct.
        """

        sequences = self.input_sequences[0,None,:]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                for Rm in self.Rm:
                    with self.subTest(modulation_rate=Rm):
                        self.mapper.set_modulation_rate(Rm)

                        mapped_sequence = self.mapper.run(sequence[None,:])

                        self.assertEqual(
                            mapped_sequence.size, int(sequence.size / Rm)
                        )
