"""
Verify LDPC encoding using MATLAB.

"""

import os
import unittest

import numpy as np
import scipy.io

from PhyPy802dot11adComponents.Codec import Encoder


class EncoderBasicTestCase(unittest.TestCase):


    def setUp(self):
        """Fetch test input data.
        """

        self.encoder = Encoder()

        basedir = os.path.abspath(os.path.dirname(__file__))

        mdict = scipy.io.loadmat( os.path.join( basedir, '_Input/sequences.mat'), appendmat=False )
        self.input_sequences = mdict['sequences'].astype('int8')

        mdict = scipy.io.loadmat( os.path.join( basedir, '_Input/encoded_sequences_1_2.mat'), appendmat=False )
        self.matlab_sequences_1_2 = mdict['encoded_sequence'].astype('int8')

        mdict = scipy.io.loadmat( os.path.join( basedir, '_Input/encoded_sequences_3_4.mat'), appendmat=False )
        self.matlab_sequences_3_4 = mdict['encoded_sequence'].astype('int8')

        mdict = scipy.io.loadmat( os.path.join( basedir, '_Input/encoded_sequences_5_8.mat'), appendmat=False )
        self.matlab_sequences_5_8 = mdict['encoded_sequence'].astype('int8')

        mdict = scipy.io.loadmat( os.path.join( basedir, '_Input/encoded_sequences_13_16.mat'), appendmat=False )
        self.matlab_sequences_13_16 = mdict['encoded_sequence'].astype('int8')


    @unittest.skip('MATLAB comparison omitted due to license expiry.')
    def test_1_2_encoding(self):
        """Verify 1/2 encoding against Matlab communication toolbox' output.
        """
        self.encoder.set_code_rate(1/2)

        for sequence_idx, sequence in enumerate(self.input_sequences):
            with self.subTest(sequence_idx=sequence_idx):
                np.testing.assert_array_equal(
                    self.encoder.run(sequence),
                    self.matlab_sequences_1_2[sequence_idx, None, :]
                )


    @unittest.skip('MATLAB comparison omitted due to license expiry.')
    def test_5_8_encoding(self):
        """Verify 5/8 encoding against Matlab communication toolbox' output.
        """
        self.encoder.set_code_rate(5/8)

        for sequence_idx, sequence in enumerate(self.input_sequences):
            with self.subTest(sequence_idx=sequence_idx):
                np.testing.assert_array_equal(
                    self.encoder.run(sequence),
                    self.matlab_sequences_5_8[sequence_idx, None, :]
                )


    @unittest.skip('MATLAB comparison omitted due to license expiry.')
    def test_3_4_encoding(self):
        """Verify 3/4 encoding against Matlab communication toolbox' output.
        """
        self.encoder.set_code_rate(3/4)

        for sequence_idx, sequence in enumerate(self.input_sequences):
            with self.subTest(sequence_idx=sequence_idx):
                np.testing.assert_array_equal(
                    self.encoder.run(sequence),
                    self.matlab_sequences_3_4[sequence_idx, None, :]
                )


    @unittest.skip('MATLAB comparison omitted due to license expiry.')
    def test_13_16_encoding(self):
        """Verify 13/16 encoding against Matlab communication toolbox' output.
        """
        self.encoder.set_code_rate(13/16)

        for sequence_idx, sequence in enumerate(self.input_sequences):
            with self.subTest(sequence_idx=sequence_idx):
                np.testing.assert_array_equal(
                    self.encoder.run(sequence),
                    self.matlab_sequences_13_16[sequence_idx, None, :]
                )
