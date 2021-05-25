import os

import numba
import numpy as np

FLOAT_BEFORE_PLUS_1 = np.nextafter( np.float(1), np.float(0) )
FLOAT_AFTER_MINUS_1 = np.nextafter( np.float(-1), np.float(0) )


@numba.jit(nopython=True)
def encode_sequence_jit( seq, H, Rc ):
    """Encode sequence."""

    # Calculate codeword size (for 7/8, take account of parity bit puncturing)
    cw_len = np.shape(H)[1] - int(Rc == 7/8) * 48

    # Calculate dataword length and dispose of decimal point (in this case: .0)
    dw_len = int(cw_len * Rc)

    # Determine the number of codewords
    N_cw = np.shape(seq)[0] / dw_len
    if N_cw % 1 != 0:
        # raise ValueError(f'Number of codewords is not int {N_cw}, for Rc {Rc}.') # Arguments must be compile-time constants
        raise ValueError('Number of codewords is not int')
    N_cw = int(N_cw)

    # Encode the sequence's datawords and join the resulting codewords
    encoded_seq = np.zeros(int(N_cw)*cw_len, dtype=np.uint8)
    for i in range(N_cw):
        dw = seq[i*dw_len:(i+1)*dw_len]
        cw = encode_dataword_jit( dw, H ) # cw is 1D ndarray
        encoded_seq[i*cw_len:(i+1)*cw_len] = cw

    return encoded_seq


@numba.jit(nopython=True)
def encode_dataword_jit( dw, H):
    """Encode dataword without repetition."""

    [M, N] = H.shape
    K = dw.size

    # Initiate parity bits and compose initial codeword
    p = np.zeros(M, dtype=np.uint8)
    cw = np.concatenate((dw, p), axis=0)

    # Bit lock - lock parity bit after setting it
    nLock = np.array([False] * K + [True] * M) # Bits marked with False are locked, the True ones can be modified

    # Iterate parity check nodes
    for m in range(M):

        row = H[m,:] # Grab current row, associated with a check node

        # Parity bit index being written in current iteration (corresponds to True value in array)
        write_idx = np.logical_and((row == 1), nLock)

        # Calculate and save parity bit
        parity = np.sum(row * cw) % 2 # Since the parity bit is always 0, it won't change the result
        cw[write_idx] = parity

        # Lock parity bit
        nLock[write_idx] = False

    return cw


@numba.jit(nopython=True)
def initialize_sequence_decoding_vars( seq, H, Rc ):
    """Common code shared between MSA and SPA decoder."""

    # Calculate codeword size
    cw_len = np.shape(H)[1]

    # Calculate dataword length and dispose of decimal point (in this case: .0)
    dw_len = int(cw_len * Rc)

    # Determine the number of codewords
    N_cw = seq.size / cw_len
    if N_cw % 1 != 0:
        raise ValueError('Number of codewords is not int.')
    N_cw = int(N_cw)

    # Decode the sequence's codewords and join the resulting datawords
    decoded_seq = np.zeros(int(N_cw * dw_len), dtype=np.uint8) # Parse to int again to prevent Numba from whining
    decoder_iterations = np.zeros(int(N_cw), dtype=np.uint8)

    return cw_len, dw_len, int(N_cw), decoded_seq, decoder_iterations


@numba.jit(nopython=True)
def decode_sequence_sum_prod_jit( seq, H, Rc, iterations, early_exit_allowed ):
    """Decode sequence using SPA."""

    cw_len, dw_len, N_cw, decoded_seq, decoder_iterations = initialize_sequence_decoding_vars( seq, H, Rc )

    for i in range(N_cw):
        cw = seq[i * cw_len:(i + 1) * cw_len]
        dw, iter = decode_codeword_sum_prod_jit(cw, H, iterations, early_exit_allowed )
        decoded_seq[i * dw_len:(i + 1) * dw_len] = dw[:dw_len]  # Discard parity bits
        decoder_iterations[i] = iter

    return decoded_seq, decoder_iterations


@numba.jit(nopython=True)
def decode_sequence_min_sum_jit( seq, H, Rc, iterations, early_exit_allowed ):
    """Decode sequence using MSA."""

    cw_len, dw_len, N_cw, decoded_seq, decoder_iterations = initialize_sequence_decoding_vars( seq, H, Rc )

    for i in range(N_cw):
        cw = seq[i * cw_len:(i + 1) * cw_len]
        dw, iter = decode_codeword_min_sum_jit(cw, H, iterations, early_exit_allowed )
        decoded_seq[i * dw_len:(i + 1) * dw_len] = dw[:dw_len]  # Discard parity bits
        decoder_iterations[i] = iter

    return decoded_seq, decoder_iterations


@numba.jit(nopython=True)
def decode_codeword_sum_prod_jit( cw, H, iterations, early_exit_allowed ):
    """Decode codeword using SPA.

    Often a RuntimeWarning: divide by zero encountered in arctanh is thrown. This is happens when +-1 is passed to the
    np.arctanh() function, which underneath implements 0.5*(log(1+x)-log(1-x)). The result is a +-inf value. Two
    mechanisms are employed to prevent further error propagation:
    - The output of np.tanh() is set to dtype=np.float128, Preventing clipping for input values below 20 (10 with float64)
    - The output of np.arctanh() is clipped to avoid the propagation of +-inf values
    """

    [M, N] = H.shape

    H = H.reshape((M*N)) # All arrays must be linear to support complex-shape indexing by Numba
    r = np.zeros(M*N) # M check node output messages (init to zero!)
    q = np.zeros(M*N) # M variable node output messages

    for iter in range(iterations):

        # Update variable node output messages (horizontal step, column-by-column)
        for m in range(M):
            for n in range(N):
                if H[m*N+n] == 0: continue # Only calculate output message if there is a connected variable node

                # Because data is 1D, first reconstruct column vector
                tmp_H_col = np.zeros(M)
                tmp_r_col = np.zeros(M)
                for i in range(M):
                    tmp_H_col[i] = H[i*N+n]
                    tmp_r_col[i] = r[i*N+n]

                # Extract input messages from check nodes coming to the current variable node (n-th column)
                idx = tmp_H_col == 1 # Extract the indexes of check nodes, connected to the variable node
                idx[m] = False # Disconnect the output message from input messages
                n_col = tmp_r_col[idx] # Extract input messages

                # Calculate the variable node output message
                q[m * N + n] = cw[n] + np.sum( n_col ) # q equal to codeword on first iteration (r = zeros)

        # Update check node output messages (horizontal step, row-by-row)
        for m in range(M):
            for n in range(N):
                if H[m*N+n] == 0: continue # Only calculate output message if there is a connected variable node

                idx = np.copy(H[m*N:(m+1)*N] == 1) # Extract the indexes of variable nodes, connected to the check node
                idx[n] = False # Disconnect current variable node (exclude output from inputs)
                m_row = q[ m*N:(m+1)*N ] # Extract m-th variable node output message
                m_row = m_row[ idx ] # Extract only the messages of connected variable nodes

                # out_msg = 2 * np.arctanh( np.prod( np.tanh(0.5 * m_row) ) ) # Calc using connected variable nodes
                prod = np.prod( np.tanh(0.5 * m_row) ) # Calc using connected variable nodes

                # By clipping to (-1, 1), bring np.arctanh() values to about [-22, 22]
                if prod == 1: prod = FLOAT_BEFORE_PLUS_1
                if prod == -1: prod = FLOAT_AFTER_MINUS_1

                out_msg = 2 * np.arctanh( prod ) # Calc using connected variable nodes

                # if abs(out_msg) == np.inf: out_msg = np.sign(out_msg) * 10**3 # Clip infinite values

                r[m*N+n] = out_msg # Calc check node output message contribution

        # Use a-priori information and combined check node outputs
        L = cw + r.reshape((M,N)).sum(axis=0)

        # Make hard decision on the output
        for i, l in enumerate(L):
            L[i] = int(l < 0) # Positive sign is zero, negative is one

        if not early_exit_allowed: continue

        # Check early exit
        mod_2_sum = 0
        # for k in range(int(672*code_rate)):
        for m in range(M): # Iterate check nodes
            check_node_sum = 0
            for i in range(672): # Iter output codeword
                check_node_sum += L[i] * H[m*N+i] # Add connected variable node contribution
            mod_2_sum += check_node_sum % 2 # Calc check node value and accumulate results
        if mod_2_sum == 0: break # Knowing all check nodes were 0 during encoding, evaluate if decoding was successful

    return L.astype(np.uint8), iter+1


@numba.jit(nopython=True)
def decode_codeword_min_sum_jit( cw, H, iterations, early_exit_allowed ):
    """Decode codeword using MSA."""

    [M, N] = H.shape

    H = H.reshape((M * N))  # All arrays must be linear to support complex-shape indexing by Numba
    r = np.zeros(M * N)  # M check node output messages (init to zero!)
    q = np.zeros(M * N)  # M variable node output messages


    for iter in range(iterations):

        # Update variable node output messages (horizontal step, column-by-column)
        for m in range(M):
            for n in range(N):
                if H[m*N+n] == 0: continue # Only calculate output message if there is a connected variable node

                # Because data is 1D, first reconstruct column vector
                tmp_H_col = np.zeros(M)
                tmp_r_col = np.zeros(M)
                for i in range(M):
                    tmp_H_col[i] = H[i*N+n]
                    tmp_r_col[i] = r[i*N+n]

                # Extract input messages from check nodes coming to the current variable node (n-th column)
                idx = tmp_H_col == 1 # Extract the indexes of check nodes, connected to the variable node
                idx[m] = False # Disconnect the output message from input messages
                n_col = tmp_r_col[idx] # Extract input messages

                # Calculate the variable node output message
                q[m * N + n] = cw[n] + np.sum( n_col ) # q equal to codeword on first iteration (r = zeros)

        # Update check node output messages (horizontal step, row-by-row)
        for m in range(M):
            for n in range(N):
                if H[m * N + n] == 0: continue  # Only calculate output message if there is a connected variable node

                idx = np.copy(H[m * N:(m + 1) * N] == 1) # Extract the indexes of variable nodes, connected to the check node
                idx[n] = False # Disconnect current variable node (exclude output from inputs)
                m_row = q[m * N:(m + 1) * N] # Extract m-th variable node output message
                m_row = m_row[idx] # Extract only the messages of connected variable nodes
                out_msg = np.prod( np.sign( m_row )) * np.min( np.abs( m_row )) # Calc using connected variable nodes

                r[m*N+n] = out_msg # Calc check node output message contribution

        # Use a-priori information and combined check node outputs
        L = cw + r.reshape((M, N)).sum(axis=0)

        # Make hard decision on the output
        for i, l in enumerate(L):
            L[i] = int(l < 0)  # Positive sign is zero, negative is one

        if not early_exit_allowed: continue

        # Check early exit
        mod_2_sum = 0
        for m in range(M):  # Iterate check nodes
            check_node_sum = 0
            for i in range(672):  # Iter output codeword
                check_node_sum += L[i] * H[m * N + i]  # Add connected variable node contribution
            mod_2_sum += check_node_sum % 2  # Calc check node value and accumulate results
        if mod_2_sum == 0: break  # Knowing all check nodes were 0 during encoding, evaluate if decoding was successful

    return L.astype(np.uint8), iter + 1


class CodecDB():
    """Codec database support class for fetching the parity check matrix."""

    def __init__(self):
        pass

    def get_parity_check_matrix( self, Rc ):
        """Get the pre-compiled parity check matrix corresponding to the chosen code rate."""
        if Rc == 1 / 2:
            filename = 'H_R_1_2.npy'
        elif Rc == 3 / 4:
            filename = 'H_R_3_4.npy'
        elif Rc == 5 / 8:
            filename = 'H_R_5_8.npy'
        elif Rc == 13 / 16:
            filename = 'H_R_13_16.npy'
        elif Rc == 7/8:
             filename = 'H_R_13_16.npy'
        else:
            raise ValueError('Unsupported code rate.', Rc)
        filepath = os.path.dirname(os.path.abspath(__file__))
        H = np.load( os.path.join( filepath, filename ))
        return H


class Codec():
    """Encoder / Decoder."""

    def __init__(self):
        """Init EncoderDecoder."""
        super().__init__()
        self.db = CodecDB()
        self.Rc = None
        self.H = None
        self.p = None

    def set_code_rate( self, Rc ):
        """Set the code rate."""
        if not ( Rc==1/2 or Rc==3/4 or Rc==5/8 or Rc==13/16 ):
            raise ValueError('Unsupported code rate selected.', Rc)
        self.Rc = Rc
        self.H = self.db.get_parity_check_matrix(Rc)

    def get_code_rate( self ):
        """Get the code rate."""
        return self.Rc


class Encoder(Codec):
    """LDPC encoder class."""

    def __init__( self ):
        """Init Encoder."""
        super().__init__()

    def encode_sequence( self, seq ):
        """Encode input sequence."""
        return encode_sequence_jit( seq, self.H, self.Rc )


class Decoder(Codec):
    """LDPC decoder class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_decoding_algorithm(self, algorithm):
        """Define the decoding algorithm."""
        if algorithm == 'SPA':
            self.decode_sequence_jit = decode_sequence_sum_prod_jit
        elif algorithm == 'MSA':
            self.decode_sequence_jit = decode_sequence_min_sum_jit
        else:
            raise(ValueError, f'Unknown algorithm {algorithm}')

    def set_max_iterations(self, value):
        """Set the maximal number of decoding iterations."""
        self.max_iterations = value

    def set_early_exit_flag(self, flag):
        """Enable or disable the usage of the decoding stop criterion."""
        self.early_exit_flag = flag

    def decode_sequence(self, seq):
        """Decode input sequence."""
        return self.decode_sequence_jit(seq, self.H, self.Rc, self.max_iterations, self.early_exit_flag)
