"""
Check data integrity after encoding and decoding.

Test all code rates and both decoding algorithms.

"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents.Codec import Encoder, Decoder


class DecoderNoRepetitionTestCase(unittest.TestCase):


    def setUp(self):
        """Fetch test input data.
        """

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_sequences = np.load( os.path.join( basedir, '_Input', 'sequences_decoder.npy') )

        # self.code_rates = [1/2, 5/8, 3/4, 13/16, 7/8]
        # self.cw_len = [672, 672, 672, 672, 624]
        # self.dw_len = [336, 420, 504, 546, 546]
        self.code_rates = [1/2, 5/8, 3/4, 13/16]
        self.cw_len = [672, 672, 672, 672]
        self.dw_len = [336, 420, 504, 546]

        self.encoder = Encoder()
        # self.encoder.set_repetition(1)
        self.encoder.set_code_rate(self.code_rates[0])

        self.decoder = Decoder()
        # self.decoder.set_repetition(1)
        self.decoder.set_code_rate(self.code_rates[0])

        self.decoding_algorithm_list = ['SPA', 'MSA']
        self.decoder.set_max_iterations(10)
        self.decoder.set_early_exit_flag(True)


    def test_encoding_decoding_loop(self):
        """Verify the encoding-decoding loop returns the initial sequence.
        """

        # sequences = self.input_sequences[0,None,:]
        sequences = self.input_sequences

        for sequence_idx, sequence in enumerate(sequences):
            with self.subTest(sequence_idx=sequence_idx):

                for decoding_algorithm_idx, decoding_algorithm in enumerate(self.decoding_algorithm_list):
                    with self.subTest(decoding_algorithm_idx=decoding_algorithm_idx):

                        self.decoder.set_decoding_algorithm(decoding_algorithm)

                        for code_rate, dw_len, cw_len in zip(self.code_rates, self.dw_len, self.cw_len):
                            with self.subTest(code_rate=code_rate):

                                self.encoder.set_code_rate(code_rate)
                                self.decoder.set_code_rate(code_rate)

                                # encoded_sequence = self.encoder.encode_sequence( sequence[None, :dw_len] )
                                encoded_sequence = self.encoder.encode_sequence( sequence[:dw_len] )
                                fake_LLR_encoded_sequence = encoded_sequence * (-2) + 1
                                decoded_sequence, _ = self.decoder.decode_sequence( fake_LLR_encoded_sequence )

                                np.testing.assert_equal(
                                    sequence[:dw_len], decoded_sequence
                                )
