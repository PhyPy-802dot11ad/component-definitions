"""
Test input/output data type and dimensions.

"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents.Codec import Encoder


class EncoderNoRepetitionBasicTestCase(unittest.TestCase):


    def setUp(self):
        """Fetch test input data.
        """

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_sequences = np.load( os.path.join( basedir, '_Input', 'sequences_encoder.npy') )

        # self.code_rates = [1/2, 5/8, 3/4, 13/16, 7/8]
        # self.cw_len = [672, 672, 672, 672, 624]
        # self.dw_len = [336, 420, 504, 546, 546]
        self.code_rates = [1/2, 5/8, 3/4, 13/16]
        self.cw_len = [672, 672, 672, 672]
        self.dw_len = [336, 420, 504, 546]

        self.encoder = Encoder()
        self.encoder.set_code_rate(self.code_rates[0])


    @unittest.skip('Data checking omitted in research package implementation.')
    def test_sequence_input_type(self):
        """Verify passing the wrong input datatypes fails.
        """

        self.encoder.set_code_rate(self.code_rates[0])

        sequences = [
            1,
            1.2,
            '1010',
            np.zeros((1, self.dw_len[0]), dtype=complex)
        ]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                with self.assertRaises(TypeError):
                    self.encoder.run(sequence)


    @unittest.skip('Data checking omited in research package implementation.')
    def test_single_dimension_input(self):
        """Verify passing a single dimension array fails.
        """

        for code_rate, dw_len in zip(self.code_rates, self.dw_len):
            with self.subTest(code_rate=code_rate):
                self.encoder.set_code_rate(code_rate)

                sequence = np.zeros(dw_len, dtype='int8')
                with self.assertRaises(ValueError):
                    self.encoder.run(sequence)


    @unittest.skip('Data checking omited in research package implementation.')
    def test_single_dataword_wrong_dimensions_input(self):
        """Verify passing a single dataword of the wrong dimensions fails.
        """

        for code_rate, dw_len in zip(self.code_rates, self.dw_len):
            with self.subTest(code_rate=code_rate):
                self.encoder.set_code_rate(code_rate)

                sequence = np.zeros((1, dw_len-1), dtype='int8')
                with self.assertRaises(ValueError):
                    self.encoder.run(sequence)

                sequence = np.zeros((1, dw_len+1), dtype='int8')
                with self.assertRaises(ValueError):
                    self.encoder.run(sequence)


    @unittest.skip('Data checking omited in research package implementation.')
    def test_multi_dataword_wrong_dimensions_input(self):
        """Verify passing a multiple datawords of the wrong dimensions fails.
        """

        for code_rate, dw_len in zip(self.code_rates, self.dw_len):
            with self.subTest(code_rate=code_rate):
                self.encoder.set_code_rate(code_rate)

                sequence = np.zeros((1, 5*dw_len-1), dtype='int8')
                with self.assertRaises(ValueError):
                    self.encoder.run(sequence)

                sequence = np.zeros((1, 5*dw_len+1), dtype='int8')
                with self.assertRaises(ValueError):
                    self.encoder.run(sequence)


    @unittest.skip('Data checking omited in research package implementation.')
    def test_output_type(self):
        """Verify an integer dtype ndarray is output.
        """

        sequences = self.input_sequences[0,None,:]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                for code_rate, dw_len in zip(self.code_rates, self.dw_len):
                    with self.subTest(code_rate=code_rate):

                        self.encoder.set_code_rate(code_rate)

                        encoded_sequence = self.encoder.run( sequence[None, :dw_len] )

                        self.assertTrue(
                            issubclass( encoded_sequence.dtype.type, np.integer )
                        )


    @unittest.skip('Data checking omited in research package implementation.')
    def test_output_dimensions(self):
        """Verify output dimensions of the codewords are correct.
        """

        sequences = self.input_sequences[0,None,:]

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):
                for code_rate, dw_len, cw_len in zip(self.code_rates, self.dw_len, self.cw_len):
                    with self.subTest(code_rate=code_rate):

                        self.encoder.set_code_rate(code_rate)

                        encoded_sequence = self.encoder.run( sequence[None, :dw_len] )

                        self.assertEqual(
                            encoded_sequence.size, cw_len
                        )
