import numpy as np


class AWGN():
    """Additive white Gaussian noise."""

    BOLTZMAN_CONSTANT = 1.38064852 * 10**(-23)
    CHANNEL_BANDWIDTH_HZ = 2 * 10**9


    def __init__(self):
        """Init AWGN."""

        super().__init__()

        self.Eb_N0_db = None
        self.noise_variance = None
        self.temperature = None


    def set_Eb_N0_db( self, Eb_N0_db ):
        """Set the desired Eb/N0 (db) value and designate it as the default means of noise power calculation."""

        self.Eb_N0_db = Eb_N0_db


    def add_noise_to_sequence( self, IQ_sequence, Rm, Rc ):
        """Add noise to input sequence."""

        noise_variance = self.__get_noise_variance_from_Eb_N0_db( IQ_sequence, Rm, Rc )
        noise_std_dev = np.sqrt(noise_variance)

        # Apply independent noise to each component
        noise = \
            np.random.normal(loc=0, scale=noise_std_dev / np.sqrt(2), size=np.shape(IQ_sequence)) + \
            1j * np.random.normal(loc=0, scale=noise_std_dev / np.sqrt(2), size=np.shape(IQ_sequence))

        noisy_IQ_sequence = IQ_sequence + noise

        return noisy_IQ_sequence, noise_variance


    def __get_noise_variance_from_Eb_N0_db( self, IQ_sequence, Rm, Rc ):
        """Get the additive white Gaussian noise standard deviation from the internal Eb/N0 (db) value."""

        Eb_N0_lin = 10 ** (self.Eb_N0_db / 10.0)

        # Energy per symbol
        E_S = np.sum(abs(IQ_sequence) ** 2) / np.size(IQ_sequence)  # Applies for both real and complex signals

        # Since noise mean is 0, the SNR denominator consists of the noise power (noise variance)
        # noise_variance = ((E_S) / (2 * Eb_N0_lin * Rc * Rm))
        noise_variance = ((E_S) / (Eb_N0_lin * Rc * Rm))

        return noise_variance

