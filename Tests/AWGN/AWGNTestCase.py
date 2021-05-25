"""
Check that the AWGN channel generates correct noise values.

"""

import os
import unittest

import numpy as np

from PhyPy802dot11adComponents import AWGN
from .GaussianProcessUtils import GaussainProcess


class AWGNTestCase(unittest.TestCase):


    def setUp(self):

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.input_sequences = np.load( os.path.join( basedir, '_Input', 'sequences.npy') )

        self.Eb_N0_db_list = [0, 0.1, 1, 10, 20]
        self.Rm = 1
        self.Rc = 1

        self.noise_variance_list = [0.1, 1, 10]

        self.temperature_list = [0, 20]
        self.signal_power_dbm_list = [10, 0, -10, -80]
        self.BOLTZMAN_CONSTANT = 1.38064852 * 10 ** (-23)
        self.CHANNEL_BANDWIDTH_HZ = 2 * 10 ** 9

        self.awgn = AWGN()


    def test_gaussian_parameters_when_using_Eb_N0_db(self):
        """Verify the AWGN variance and mean are correct when using Eb/N0 ratio."""

        gaussian_process = GaussainProcess()

        input_sequences = self.input_sequences


        for Eb_N0_db in self.Eb_N0_db_list:
            with self.subTest(Eb_N0_db=Eb_N0_db):

                self.awgn.set_Eb_N0_db(Eb_N0_db)

                variance_avg = .0
                mean_avg = .0

                for sequence_idx, sequence in enumerate(input_sequences):
                    with self.subTest(sequence=sequence):

                        noisy_sequence, _ = self.awgn.add_noise_to_sequence(sequence, self.Rm, self.Rc)
                        gaussian_process.set_sequence(noisy_sequence)
                        gaussian_process.calc_and_update_mean_and_variance()

                        variance_avg += gaussian_process.variance
                        mean_avg += gaussian_process.mean

                expected_variance = 10 ** (-Eb_N0_db/10) / 2 * (1+1j)
                expected_mean = (1+0j)

                variance_avg /= (sequence_idx+1)
                mean_avg /= (sequence_idx+1)

                self.assertAlmostEqual( expected_variance, variance_avg, delta=0.1)
                self.assertAlmostEqual( expected_mean, mean_avg, delta=0.1)


    @unittest.skip('Noise based on direct variance input omitted.')
    def test_gaussian_parameters_when_using_noise_variance(self):
        """Verify the AWGN variance and mean are correct when using a fixed noise power value."""

        gaussian_process = GaussainProcess()

        input_sequences = self.input_sequences

        for noise_variance in self.noise_variance_list:
            with self.subTest(noise_variance=noise_variance):

                self.awgn.set_noise_variance(noise_variance)

                variance_avg = .0
                mean_avg = .0

                for sequence_idx, sequence in enumerate(input_sequences):
                    with self.subTest(sequence=sequence):

                        noisy_sequence = self.awgn.add_noise(sequence[None,:])
                        gaussian_process.set_sequence(noisy_sequence)
                        gaussian_process.calc_and_update_mean_and_variance()

                        variance_avg += gaussian_process.variance
                        mean_avg += gaussian_process.mean

                expected_variance = noise_variance / 2 * (1+1j)
                expected_mean = (1+0j)

                variance_avg /= (sequence_idx+1)
                mean_avg /= (sequence_idx+1)

                self.assertAlmostEqual( expected_variance, variance_avg, delta=0.1)
                self.assertAlmostEqual( expected_mean, mean_avg, delta=0.1)


    @unittest.skip('Temperature-based noise omitted.')
    def test_gaussian_parameters_when_using_temperature(self):
        """Verify the AWGN variance and mean are correct when using the temperature as an input."""

        gaussian_process = GaussainProcess()

        input_sequences = self.input_sequences

        for temperature in self.temperature_list:
            with self.subTest(temperature=temperature):

                for signal_power_dbm in self.signal_power_dbm_list:
                    with self.subTest(signal_power_dbm=signal_power_dbm):

                        self.awgn.set_temperature(temperature)

                        variance_avg = .0
                        mean_avg = .0

                        for sequence_idx, sequence in enumerate(input_sequences):
                            with self.subTest(sequence=sequence):

                                noisy_sequence = self.awgn.add_noise(sequence[None,:], signal_power_dbm=signal_power_dbm)
                                gaussian_process.set_sequence(noisy_sequence)
                                gaussian_process.calc_and_update_mean_and_variance()

                                variance_avg += gaussian_process.variance
                                mean_avg += gaussian_process.mean

                        # Calculate the noise power in dBm
                        noise_power_dbm = \
                            10 * np.log10(self.BOLTZMAN_CONSTANT * self.CHANNEL_BANDWIDTH_HZ * 1000) + \
                            10 * np.log10(273 + temperature)

                        # Calculate linear signal to noise ratio
                        SNR = 10 ** ((signal_power_dbm - noise_power_dbm)/10)

                        # Calculate the energy per symbol (signal power)
                        Pn = 1

                        # Calculate noise power (variance)
                        noise_variance = Pn / SNR

                        expected_variance = noise_variance / 2 * (1+1j)
                        expected_mean = (1+0j)

                        variance_avg /= (sequence_idx+1)
                        mean_avg /= (sequence_idx+1)

                        self.assertAlmostEqual( expected_variance, variance_avg, delta=0.1)
                        self.assertAlmostEqual( expected_mean, mean_avg, delta=0.1)


