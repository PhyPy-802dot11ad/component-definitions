"""
Check data integrity after mapping and demapping.

Test all modulation rates and demapping algorithms.

"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents.Madem import Mapper, Demapper


class DemodulatorAdvancedTestCase(unittest.TestCase):

    def setUp(self):

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_binary_sequences = np.load( os.path.join( basedir, '_Input', 'binary_sequences_demapper.npy') )

        self.Rm = [1, 2, 4, 6]
        self.demapping_algorithm_list = ['optimal', 'suboptimal', 'decision threshold']
        self.noise_variance = 0.1

        self.mapper = Mapper()
        self.demapper = Demapper()


    def test_mapping_demapping_loop(self):
        """Verify the mapping-demapping loop returns the initial sequence.
        """

        for Rm in self.Rm:
            with self.subTest(modulation_rate=Rm):

                self.mapper.set_modulation_rate(Rm)
                self.demapper.set_modulation_rate(Rm)

                for demapping_algorithm_idx, demapping_algorithm in enumerate(self.demapping_algorithm_list):
                    with self.subTest(demapping_algorithm_idx=demapping_algorithm_idx):

                        self.demapper.set_demapping_algorithm(demapping_algorithm)

                        for binary_sequence_idx, binary_sequence in enumerate(self.input_binary_sequences):
                            with self.subTest(binary_sequence_idx=binary_sequence_idx):

                                mapped_sequence = self.mapper.map_sequence( binary_sequence )
                                demapped_sequence = self.demapper.demap_sequence( mapped_sequence, self.noise_variance )

                                fake_hard_decoding = (demapped_sequence < 0).astype('int8')

                                np.testing.assert_array_equal(
                                    fake_hard_decoding, binary_sequence
                                )
