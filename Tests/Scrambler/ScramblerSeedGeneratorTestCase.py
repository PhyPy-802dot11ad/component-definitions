"""ScramblerSeeder class test case.
"""

import unittest

import numpy as np

# from phy802dot11ad.Component.Scrambler.ScramblerSeedGenerator import ScramblerSeedGenerator
from PhyPy802dot11adComponents.Scrambler import ScramblerSeedGenerator


class ScramblerSeederTestCase(unittest.TestCase):

    def setUp(self):
        """Init internal ScramblerSeedGenerator.
        """

        self.seed_generator = ScramblerSeedGenerator()

    @unittest.skip('Data checking omitted in research package implementation.')
    def test_x7x6_input_type(self):
        """Verify the class raises a TypeError for incorrect x7x6 input data types.
        """

        x7x6s = [
            1,
            1.2,
            '1010',
            np.zeros((1,2), dtype=complex)
        ]

        for x7x6_idx, x7x6 in enumerate(x7x6s):
            with self.subTest(x7x6_idx=x7x6_idx):
                with self.assertRaises(TypeError):
                    self.seed_generator.generate_pseudorandom_seed(x7x6)


    def test_x7x6_input_dimensions(self):
        """Verify the class raises a ValueError for incorrect x7x6 input dimensions.
        """

        x7x6s = [
            np.zeros(7, dtype='int8'),
            np.zeros((1, 8), dtype='int8'),
            np.zeros((1, 6), dtype='int8'),
            np.zeros((2, 7), dtype='int8')
        ]

        for x7x6_idx, x7x6 in enumerate(x7x6s):
            with self.subTest(x7x6_idx=x7x6_idx):
                with self.assertRaises(ValueError):
                    self.seed_generator.generate_pseudorandom_seed(x7x6)


    def test_pseudorandom_seed_generation(self):
        """Verify the generated pseudorandom sequence is non-zero.
        """

        x7x6_arr = [
            np.array([[1,0]]),
            None
        ]

        for x7x6_idx, x7x6 in enumerate(x7x6_arr):
            with self.subTest( x7x6_idx=x7x6_idx ):
                seed = self.seed_generator.generate_pseudorandom_seed()
                self.assertGreater( np.sum(seed), 0 )