import numba
import numpy as np


xor = lambda x, y: int(bool(x) != bool(y))


@numba.jit(nopython=True)
def xor_jit(x, y):
    return np.uint8(x != y)


@numba.jit(nopython=True)
def scramble_sequence_jit(sequence, register_value):
    """Scramble sequence."""

    for i, bit in enumerate(sequence):
        x7_xor_x4 = xor_jit(register_value[3], register_value[6]) # Internal scrambler register XoR
        sequence[i] = xor_jit(x7_xor_x4, bit) # XoR between sequence and scrambler register
        register_value[1:] = register_value[:-1] # Shift
        register_value[0] = x7_xor_x4 # Shift (set first to XoR result)

    return sequence.astype(np.uint8), register_value # Even if the input isn't, the output will always be uint8


class ScramblerSeedGenerator():
    """Class responsible for creating a pseudorandom seed sequence for the Scrambler's internal 7-bit register."""

    def __init__(self):
        pass

    def generate_pseudorandom_seed(self, x7x6=None):
        """Generate pseudorandom sequence for seeding the scrambler's register."""

        seed = np.random.randint(0, 2, 7, dtype=np.uint8)  # Generate random sequence

        # If X7 and X6 are supplied, include them in the generated seed array
        if not (x7x6 is None):
            seed[5:] = x7x6[::-1]  # Switch direction (x7x6 -> x6x7)

        # At least one element must be non-zero
        if seed.max() == 0:
            seed[np.random.randint(0, 7)] = 1  # Set element at random index to 1

        return seed.astype(np.uint8)


class Scrambler():
    """7-bit scrambler component used for header and data scrambling.
    """

    def __init__(self):
        """Init Scrambler."""
        super().__init__()
        self.seed_generator = ScramblerSeedGenerator()
        self.register = self.seed_generator.generate_pseudorandom_seed()

    def get_register( self ):
        return np.copy(self.register)

    def set_register( self, register ):
        """Set the register to a new value."""
        self.register = np.copy(register)

    def generate_pseudorandom_seed( self, *args, **kwargs ):
        return self.seed_generator.generate_pseudorandom_seed( *args, **kwargs )

    def scramble( self, sequence ):
        """Scramble sequence."""
        seq, self.register = scramble_sequence_jit(np.copy(sequence), self.register)
        return seq
