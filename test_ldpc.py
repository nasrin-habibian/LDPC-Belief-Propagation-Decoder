import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ldpc import (
    belief_propagation_decode,
    bit_flip_decode,
    generate_gallager_ldpc,
    syndrome,
)


def test_gallager_matrix_is_regular():
    H = generate_gallager_ldpc(n=96, variable_degree=3, check_degree=6, seed=1)

    assert H.shape == (48, 96)
    assert np.all(H.sum(axis=0) == 3)
    assert np.all(H.sum(axis=1) == 6)


def test_zero_codeword_has_zero_syndrome():
    H = generate_gallager_ldpc(seed=1)
    zero = np.zeros(H.shape[1], dtype=np.int8)

    assert np.all(syndrome(H, zero) == 0)


def test_bp_decodes_a_low_noise_word():
    H = generate_gallager_ldpc(seed=1)
    received = np.zeros(H.shape[1], dtype=np.int8)

    # Introduce one error.
    received[10] = 1

    estimate, ok, _ = belief_propagation_decode(
        H, received, crossover_probability=0.02, max_iterations=50
    )

    assert ok
    assert np.all(estimate == 0)


def test_bit_flip_returns_a_binary_word():
    H = generate_gallager_ldpc(seed=1)
    received = np.zeros(H.shape[1], dtype=np.int8)
    received[10] = 1

    estimate, ok, _ = bit_flip_decode(H, received, max_iterations=30)

    assert estimate.shape == received.shape
    assert np.all((estimate == 0) | (estimate == 1))
    assert isinstance(ok, (bool, np.bool_))
