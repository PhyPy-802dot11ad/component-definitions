"""
Check proper constellation and norm factor are generated.

Only 16QAM test implemented.

"""

import unittest

import numpy as np

from PhyPy802dot11adComponents.Madem import Mapper


class MapperAdvancedTestCase(unittest.TestCase):


    def setUp(self):
        self.mapper = Mapper()


    def test_16QAM_constellation_and_norm_factor(self):
        """Verify the 16QAM constellation and norm factor get loaded and that they are equal to those in the standard.
        """

        test_constellation = generate_16QAM_constellation()
        test_norm_factor = np.array([1 / np.sqrt(10)])

        self.mapper.set_modulation_rate(4)
        constellation = self.mapper.get_constellation()
        norm_factor = self.mapper.get_norm_factor()

        np.testing.assert_array_equal( constellation, test_constellation )
        self.assertEqual( norm_factor, test_norm_factor )



def generate_16QAM_constellation():
    """Generate 16QAM constellation as in standard.

    :return:
    :rtype:
    """
    Rm = 4

    seq = generate_binary_sequence(Rm)

    constellation = np.array([])
    for k in range(0, 16):
        tmp = np.array(0 + 0j)
        tmp.real = (4 * seq[k, 0] - 2) - (2 * seq[k, 0] - 1) * (2 * seq[k, 1] - 1)
        tmp.imag = (4 * seq[k, 2] - 2) - (2 * seq[k, 2] - 1) * (2 * seq[k, 3] - 1)

        tmp = np.round(tmp, decimals=12)

        constellation = np.concatenate((constellation, [tmp]), axis=0)

    return constellation


def generate_binary_sequence(Rm):
    """
    Based on the modulation rate, generate binary sequence covering all possible constellation points.

    :param Rm: Modulation rate.
    :type Rm: int
    :return: Binary sequence covering all avalable constellation points.
    :rtype: 2D int np.array
    """
    len = 2 ** Rm

    seq = np.zeros((len, Rm), dtype=int)
    for k in range(0, len):

        # bin_str = f"{k:04b}"
        bin_str = f"{k:0{Rm}b}"

        for i, char in enumerate(bin_str):
            seq[k, i] = int(char)

    return seq
