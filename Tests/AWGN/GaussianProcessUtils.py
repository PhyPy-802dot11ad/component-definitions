"""Source file containing Gaussian process operations.
"""

import numpy as np


class GaussainProcessComparator():

    def __init__( self, mean_tolerace=None, variance_tolerace=None ):
        """Init GaussianProcessComparator.

        :param mean_tolerace: Tolerance for the mean discrepancy between the two compared values.
        :type mean_tolerace: int or float
        :param variance_tolerace: Tolerance for the variance discrepancy between the two compared values.
        :type variance_tolerace: int or float
        """
        self.set_mean_tolerance(mean_tolerace)
        self.set_variance_tolerance(variance_tolerace)

    def set_mean_tolerance( self, mean_tolerance ):
        """Set the maximal allowed tolerance for the mean discrepancy between the two compared values.

        :param mean_tolerace: Tolerance for the mean discrepancy between the two compared values.
        :type mean_tolerace: int or float
        """
        self.mean_tolerance = mean_tolerance

    def set_variance_tolerance( self, variance_tolerace ):
        """Set the maximal allowed tolerance for the mean discrepancy between the two compared values.

        :param variance_tolerace: Tolerance for the variance discrepancy between the two compared values.
        :type variance_tolerace: int or float
        """
        self.variance_tolerance = variance_tolerace

    def are_processes_equal( self, process1, process2 ):
        """

        :param process1: Gaussian process number 1.
        :type process1: GaussainProcess
        :param process2: Gaussian process number 1.
        :type process2: GaussainProcess
        :return: Results of mean and variance equality evaluation
        :rtype: ndaray
        """
        mean_equality = True
        variance_equality = True

        mean_discrepancy = process1.mean - process2.mean
        for component in [mean_discrepancy.real, mean_discrepancy.imag]:
            if abs(component) > self.mean_tolerance:
                mean_equality = False

        variance_quotient = \
            process1.variance.real / process2.variance.real + \
            1j * (process1.variance.imag / process2.variance.imag)

        for component in [variance_quotient.real, variance_quotient.imag]:
            if abs(component-1) > self.variance_tolerance:
                variance_equality = False

        return np.array([mean_equality, variance_equality])



class GaussainProcess():


    def __init__(self):
        pass


    def set_sequence( self, sequence ):
        """Set the Gaussian process' data sequence.

        :param sequence: Data sequence (1,N)
        :type sequence: ndarray
        :return: Object for the purpose of method chaining
        :rtype: object
        """
        self.sequence = sequence
        return self


    def calc_and_update_mean_and_variance( self ):
        """Calculate the mean and variance parameters of the Gaussian process and store them internally."""
        seq = self.sequence

        len = seq.size

        mean = sum(seq) / len

        variance = 0.
        for i in range(0,len):
            variance += (mean.real - seq[i].real)**2 + 1j*(mean.imag - seq[i].imag)**2
        variance /= len

        self.mean = mean
        self.variance = variance
