"""
Mapper and demapper implementations, according to IEEE 802.11ad.

Optimal, sub-optimal, and decision threshold demappers are implemented.

"""

import os

import numpy as np
import numba


MIN_POSITIVE_FLOAT64 = np.nextafter(np.float32(0), np.float32(1)) # 1e-45
MAX_POSITIVE_FLOAT64 = np.nextafter(np.inf, np.float32(1)) # 1.7976931348623157e+308


@numba.jit(nopython=True)
def normalize_jit(seq, norm_factor):
    """De-Normalize the modulated sequence using the scheme's corresponding normalization factor."""
    normalized_sequence = seq * norm_factor
    return normalized_sequence

@numba.jit(nopython=True)
def rotate_PI_2_jit(seq, rotation_k):
    """Rotate sequence by PI/2*k forwards, where k is the sequence element index."""
    len = np.shape(seq)[0]
    for k in range(0, len):
        seq[k] *= np.exp(1j * np.pi * rotation_k / 2.0)
        rotation_k += 1
    # Round off numerical error
    # np.round(seq, decimals=12, out=seq)
    np.round(seq, 12, seq) # Args instead of kwargs to keep Numba from complainig
    return seq, rotation_k

@numba.jit(nopython=True)
def map_to_constellation_jit(seq, Rm, constellation):
    """Map the binary sequence to the modulation constellation."""

    N_s = seq.size / Rm
    if N_s % 1 != 0:
        raise ValueError('Number of symbols is not int.')
    N_s = int(N_s)

    mapped_seq = np.zeros(int(N_s), dtype=np.complex64) # Re-cast to avoid Numba complaints

    for i in range(N_s):
        sub_seq_bin = seq[i * Rm:(i + 1) * Rm]
        # sub_seq_dec = np.dot(sub_seq_bin, 1 << np.arange(sub_seq_bin.size)[::-1]) # Numba supports only 1-arg form
        sub_seq_dec = 0
        for k in range(Rm):
            sub_seq_dec += sub_seq_bin[Rm-1-k] * (2**k) # Dot product substitute (get decimal constellation point index)
        constellation_point = constellation[sub_seq_dec]
        mapped_seq[i] = constellation_point

    return mapped_seq


@numba.jit(nopython=True)
def de_normalize_jit(seq, norm_factor):
    """De-Normalize the modulated sequence using the scheme's corresponding normalization factor."""
    denormalized_sequence = seq / norm_factor
    return denormalized_sequence

@numba.jit(nopython=True)
def de_rotate_PI_2_jit(seq, rotation_k):
    """Rotate sequence by PI/2*k backwards, where k is the sequence element index."""
    len = np.shape(seq)[0]
    for k in range(0, len):
        seq[k] *= np.exp(-1j * np.pi * rotation_k / 2.0)
        rotation_k += 1
    return seq, rotation_k


@numba.jit(nopython=True)
def optimal_demap_sequence_from_constellation(r_seq, noise_var, Rm, constellation):
    """Demap symbol sequence using optimal algorithm."""
    bit_llr_values = np.zeros(r_seq.size*Rm)
    for i, r in enumerate(r_seq):
        bit_llr_values[i*Rm:(i+1)*Rm] = optimal_demap_symbol_from_constellation(r, noise_var, Rm, constellation)
    return bit_llr_values

@numba.jit(nopython=True)
def optimal_demap_symbol_from_constellation(r, noise_var, Rm, constellation):
    """Demap single symbol using optimal algorithm."""

    llr = np.zeros(Rm)

    for k in range(Rm):
        sum_one = sum_zero = 0
        for i, c in enumerate(constellation):

            exponent = abs(r - c)**2 / (2*noise_var)

            val = np.exp(-exponent)

            if (i >> k) & 1:
                sum_one += val
            else:
                sum_zero += val

        if sum_one == 0: sum_one = MIN_POSITIVE_FLOAT64 # Avoid log of zero
        if sum_zero == 0: sum_zero = MIN_POSITIVE_FLOAT64 # Avoid division by zero

        # llr[k] = np.log( sum_one / sum_zero )
        llr[k] = np.log( sum_zero / sum_one )

    return llr[::-1]


@numba.jit(nopython=True)
def suboptimal_demap_sequence_from_constellation(r_seq, noise_var, Rm, constellation):
    """Demap symbol sequence using sub-optimal algorithm."""
    bit_llr_values = np.zeros(r_seq.size*Rm)
    for i, r in enumerate(r_seq):
        bit_llr_values[i*Rm:(i+1)*Rm] = suboptimal_demap_symbol_from_constellation(r, noise_var, Rm, constellation)
    return bit_llr_values

@numba.jit(nopython=True)
def suboptimal_demap_symbol_from_constellation(r, noise_var, Rm, constellation):
    """Demap single symbol using sub-optimal algorithm."""

    llr = np.zeros(Rm)

    for k in range(Rm):

        dist_zero = []
        dist_one = []

        for i, c in enumerate(constellation):

            dist = abs(r - c)**2

            if i >> k & 1:
                dist_one.append(dist)
            else:
                dist_zero.append(dist)

        # llr[k] = (min(dist_zero) - min(dist_one)) / (2*noise_var)
        llr[k] = (min(dist_one) - min(dist_zero)) / (2*noise_var)

    return llr[::-1]


@numba.jit(nopython=True)
def decision_threshold_demap_sequence_from_constellation(r_seq, noise_var, Rm, _):
    """Demap symbol sequence using decision threshold algorithm."""
    if Rm == 1: return decision_threshold_BPSK_demap_sequence_from_constellation( r_seq, noise_var )
    elif Rm == 2: return decision_threshold_QPSK_demap_sequence_from_constellation( r_seq, noise_var )
    elif Rm == 4: return decision_threshold_16QAM_demap_sequence_from_constellation( r_seq, noise_var )
    elif Rm == 6: return decision_threshold_64QAM_demap_sequence_from_constellation( r_seq, noise_var )

@numba.jit(nopython=True)
def decision_threshold_BPSK_demap_sequence_from_constellation(r_seq, noise_var):
    """Demap BPSK symbol sequence using decision threshold algorithm."""
    bit_llr_values = np.zeros(r_seq.size)
    for i, r in enumerate(r_seq):
        bit_llr_values[i] = decision_threshold_BPSK_demap_symbol_from_constellation(r, noise_var)
    return bit_llr_values

@numba.jit(nopython=True)
def decision_threshold_QPSK_demap_sequence_from_constellation(r_seq, noise_var):
    """Demap QPSK symbol sequence using decision threshold algorithm."""
    bit_llr_values = np.zeros(r_seq.size*2)
    for i, r in enumerate(r_seq):
        bit_llr_values[i*2:(i+1)*2] = decision_threshold_QPSK_demap_symbol_from_constellation(r, noise_var)
    return bit_llr_values

@numba.jit(nopython=True)
def decision_threshold_16QAM_demap_sequence_from_constellation(r_seq, noise_var):
    """Demap 16QAM symbol sequence using decision threshold algorithm."""
    bit_llr_values = np.zeros(r_seq.size*4)
    for i, r in enumerate(r_seq):
        bit_llr_values[i*4:(i+1)*4] = decision_threshold_16QAM_demap_symbol_from_constellation(r, noise_var)
    return bit_llr_values

@numba.jit(nopython=True)
def decision_threshold_64QAM_demap_sequence_from_constellation(r_seq, noise_var):
    """Demap 64QAM symbol sequence using decision threshold algorithm."""
    bit_llr_values = np.zeros(r_seq.size*6)
    for i, r in enumerate(r_seq):
        bit_llr_values[i*6:(i+1)*6] = decision_threshold_64QAM_demap_symbol_from_constellation(r, noise_var)
    return bit_llr_values

@numba.jit(nopython=True)
def decision_threshold_demap_symbol_from_constellation(r, noise_var, Rm, _):
    """Demap single symbol using decision threshold algorithm."""
    if Rm == 1: return decision_threshold_BPSK_demap_symbol_from_constellation( r, noise_var )
    elif Rm == 2: return decision_threshold_QPSK_demap_symbol_from_constellation( r, noise_var )
    elif Rm == 4: return decision_threshold_16QAM_demap_symbol_from_constellation( r, noise_var )
    elif Rm == 6: return decision_threshold_64QAM_demap_symbol_from_constellation( r, noise_var )


@numba.jit(nopython=True)
def decision_threshold_BPSK_demap_symbol_from_constellation(r, noise_var):
    # llr = np.zeros(1)
    # llr[0] = r.real / (2*noise_var)
    llr = r.real / (2*noise_var)
    return -llr

@numba.jit(nopython=True)
def decision_threshold_QPSK_demap_symbol_from_constellation(r, noise_var):
    llr = np.zeros(2)
    llr[0] = (r.real - r.imag) / (2*noise_var)
    llr[1] = (r.real + r.imag) / (2*noise_var)
    return -llr

@numba.jit(nopython=True)
def decision_threshold_16QAM_demap_symbol_from_constellation(r, noise_var):
    llr = np.zeros(4)
    llr[0] = r.real / (2*noise_var)
    llr[1] = ( 2 - abs(r.real) ) / (2*noise_var)
    llr[2] = r.imag / (2*noise_var)
    llr[3] = ( 2 - abs(r.imag) ) / (2*noise_var)
    return -llr

@numba.jit(nopython=True)
def decision_threshold_64QAM_demap_symbol_from_constellation(r, noise_var):
    llr = np.zeros(6)
    llr[0] = r.real / (2*noise_var)
    llr[1] = ( 4 - abs(r.real) ) / (2*noise_var)
    llr[2] = ( -abs(4 - abs(r.real)) + 2 ) / (2*noise_var)
    llr[3] = r.imag / (2*noise_var)
    llr[4] = ( 4 - abs(r.imag) ) / (2*noise_var)
    llr[5] = ( -abs(4 - abs(r.imag)) + 2 ) / (2*noise_var)
    return -llr


class MapperDemapperDB():

    def __init__(self):
        pass

    def get_constellation( self, Rm ):
        """Fetch the constellation and norm factor from the associated file."""

        if not (Rm == 1 or Rm == 2 or Rm == 4 or Rm == 6 ):
            raise ValueError( 'Incorrect modulation rate', Rm )

        # Constellation files are stored in the same directory
        filepath = os.path.dirname(os.path.abspath(__file__))
        filename = 'constellation_%d.npz' % Rm

        npz = np.load( os.path.join( filepath, filename ))

        return [ npz['constellation'], npz['nf'] ]


class Madem():
    """Mapper / demapper superclass."""

    def __init__(self):
        self.db = MapperDemapperDB()
        self.Rm = None
        self.constellation = None
        self.norm_factor = None

    def set_modulation_rate( self, Rm ):
        """Set the modulation rate and fetch the constellation and norm factor."""
        if not (Rm == 1 or Rm == 2 or Rm == 4 or Rm == 6 ):
            raise ValueError( 'Incorrect modulation rate', Rm )
        self.Rm = Rm
        [ self.constellation, self.norm_factor ] = self.db.get_constellation(Rm)

    def get_constellation(self):
        """Return the currently loaded constellation."""
        return self.constellation

    def get_norm_factor(self):
        """Return the norm factor associated with the currently loaded constellation."""
        return self.norm_factor

    def reset_rotation_k(self):
        """Reset the rotation back to 0, making the demapper ready to process a new sequence."""
        self.rotation_k = 0


class Mapper(Madem):
    """Mapper definition."""

    def __init__(self):
        super().__init__()
        self.norm_factor = None
        self.reset_rotation_k()

    def map_to_constellation(self, seq):
        """Map the binary sequence to the modulation constellation."""
        return map_to_constellation_jit(seq, self.Rm, self.constellation)

    def rotate_PI_2(self, seq):
        """Rotate sequence by PI/2*k forwards, where k is the sequence element index."""
        seq, self.rotation_k = rotate_PI_2_jit(seq, self.rotation_k)
        return seq

    def normalize(self, seq):
        """Normalize the modulated sequence using the scheme's corresponding normalization factor."""
        return normalize_jit(seq, self.norm_factor)

    def map_sequence(self, seq):
        """Map sequence."""
        self.reset_rotation_k()
        mapped_seq = self.map_to_constellation(seq)
        # mapped_seq = self.rotate_PI_2(mapped_seq)
        # mapped_seq = self.normalize(mapped_seq)
        return mapped_seq

    def map_symbol(self, seq):
        """Map symbol."""
        mapped_seq = self.map_to_constellation(seq)
        mapped_seq = self.rotate_PI_2(mapped_seq)
        mapped_seq = self.normalize(mapped_seq)
        return mapped_seq


class Demapper(Madem):
    """Demapper definition."""

    def __init__(self):
        super().__init__()
        self.reset_rotation_k()

    def set_demapping_algorithm(self, algorithm):
        """"Set the demapper to use one of three algorithms."""
        if algorithm == 'optimal':
            self.demap_sequence_from_constellation = optimal_demap_sequence_from_constellation
            self.demap_symbol_from_constellation = optimal_demap_symbol_from_constellation
        elif algorithm == 'suboptimal':
            self.demap_sequence_from_constellation = suboptimal_demap_sequence_from_constellation
            self.demap_symbol_from_constellation = suboptimal_demap_symbol_from_constellation
        elif algorithm == 'decision threshold':
            self.demap_sequence_from_constellation = decision_threshold_demap_sequence_from_constellation
            self.demap_symbol_from_constellation = decision_threshold_demap_symbol_from_constellation
        else: raise ValueError (f'Unknown algorithm: {algorithm}')

    def de_rotate_PI_2(self, seq):
        """"De-rotate sequence or symbol."""
        seq, self.rotation_k = de_rotate_PI_2_jit(seq, self.rotation_k)
        return seq

    def de_normalize(self, seq):
        """"De-normalize sequence or symbol."""
        return de_normalize_jit(seq, self.norm_factor)

    def demap_sequence(self, r_seq, noise_var):
        """"De-normalize, de-rotate and de-demap sequence."""
        self.reset_rotation_k()
        # r_seq = self.de_normalize(r_seq)
        # r_seq = self.de_rotate_PI_2(r_seq)
        llr = self.demap_sequence_from_constellation(r_seq, noise_var, self.Rm, self.constellation)
        return llr

    def demap_symbol(self, r_seq, noise_var):
        """"De-normalize, de-rotate and de-demap symbol. K reset is done manually on demand."""
        r_seq = self.de_normalize(r_seq)
        r_seq = self.de_rotate_PI_2(r_seq)
        llr = self.demap_symbol_from_constellation(r_seq, noise_var, self.Rm, self.constellation)
        return llr
