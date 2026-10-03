from __future__ import annotations

import numpy as np


def generate_gallager_ldpc(
    n: int = 96,
    variable_degree: int = 3,
    check_degree: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Generate a regular Gallager-style LDPC parity-check matrix.

    The returned matrix H has shape (m, n), with constant column degree
    ``variable_degree`` and constant row degree ``check_degree``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if variable_degree <= 0 or check_degree <= 0:
        raise ValueError("Degrees must be positive.")
    if (n * variable_degree) % check_degree != 0:
        raise ValueError("n * variable_degree must be divisible by check_degree.")
    if check_degree > n:
        raise ValueError("check_degree cannot exceed n.")

    m = (n * variable_degree) // check_degree
    rows_per_block = m // variable_degree

    # Base block: every row contains check_degree consecutive ones.
    base = np.zeros((rows_per_block, n), dtype=np.int8)
    for row in range(rows_per_block):
        start = row * check_degree
        base[row, start:start + check_degree] = 1

    rng = np.random.default_rng(seed)
    blocks = [base]

    # Independently permute columns in each additional block.
    for _ in range(1, variable_degree):
        permutation = rng.permutation(n)
        blocks.append(base[:, permutation])

    H = np.vstack(blocks)

    # Sanity checks for a regular matrix.
    assert np.all(H.sum(axis=0) == variable_degree)
    assert np.all(H.sum(axis=1) == check_degree)

    return H


def transmit_bsc(
    codeword: np.ndarray,
    crossover_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Transmit a binary codeword through a BSC.

    Returns:
        received: received binary vector
        errors: binary error vector
    """
    if not 0.0 < crossover_probability < 0.5:
        raise ValueError("crossover_probability must be between 0 and 0.5.")

    codeword = np.asarray(codeword, dtype=np.int8)
    errors = (rng.random(codeword.size) < crossover_probability).astype(np.int8)
    received = np.bitwise_xor(codeword, errors)
    return received, errors


def syndrome(H: np.ndarray, word: np.ndarray) -> np.ndarray:
    """Return Hx mod 2."""
    return (H @ word % 2).astype(np.int8)


def bit_flip_decode(
    H: np.ndarray,
    received: np.ndarray,
    max_iterations: int = 30,
) -> tuple[np.ndarray, bool, int]:
    """Simple parallel hard-decision bit-flipping decoder.

    At each iteration, variables participating in more unsatisfied checks
    than half their degree are flipped.
    """
    H = np.asarray(H, dtype=np.int8)
    x = np.asarray(received, dtype=np.int8).copy()

    variable_degree = H.sum(axis=0)
    if not np.all(variable_degree == variable_degree[0]):
        raise ValueError("bit_flip_decode currently expects a regular LDPC matrix.")

    for iteration in range(1, max_iterations + 1):
        syn = syndrome(H, x)
        if not np.any(syn):
            return x, True, iteration

        unsatisfied_count = H.T @ syn
        threshold = variable_degree // 2 + 1
        flips = unsatisfied_count >= threshold

        if not np.any(flips):
            return x, False, iteration

        x[flips] ^= 1

    return x, bool(not np.any(syndrome(H, x))), max_iterations


def _build_graph(H: np.ndarray):
    """Build check-to-variable and variable-to-check adjacency lists."""
    check_to_vars = [np.flatnonzero(H[c]).tolist() for c in range(H.shape[0])]
    var_to_checks = [np.flatnonzero(H[:, v]).tolist() for v in range(H.shape[1])]
    return check_to_vars, var_to_checks


def belief_propagation_decode(
    H: np.ndarray,
    received: np.ndarray,
    crossover_probability: float,
    max_iterations: int = 30,
    llr_clip: float = 20.0,
) -> tuple[np.ndarray, bool, int]:
    """Sum-product belief-propagation decoder for a BSC.

    Uses LLR-domain messages and early stopping when the estimated word
    satisfies all parity checks.
    """
    H = np.asarray(H, dtype=np.int8)
    y = np.asarray(received, dtype=np.int8)

    if not 0.0 < crossover_probability < 0.5:
        raise ValueError("crossover_probability must be between 0 and 0.5.")
    if y.size != H.shape[1]:
        raise ValueError("received length must match H's number of columns.")

    check_to_vars, var_to_checks = _build_graph(H)

    channel_llr = np.log((1.0 - crossover_probability) / crossover_probability)
    llr = np.where(y == 0, channel_llr, -channel_llr).astype(float)

    # Messages are stored in dictionaries keyed by (check, variable).
    q = {}
    r = {}

    for c, variables in enumerate(check_to_vars):
        for v in variables:
            q[(v, c)] = llr[v]
            r[(c, v)] = 0.0

    for iteration in range(1, max_iterations + 1):
        # Check-node update.
        for c, variables in enumerate(check_to_vars):
            for v in variables:
                product = 1.0
                for u in variables:
                    if u == v:
                        continue
                    product *= np.tanh(np.clip(q[(u, c)], -llr_clip, llr_clip) / 2.0)

                product = np.clip(product, -0.999999999, 0.999999999)
                r[(c, v)] = 2.0 * np.arctanh(product)

        # Variable-node update and posterior LLRs.
        posterior = np.empty(H.shape[1], dtype=float)
        for v, checks in enumerate(var_to_checks):
            posterior[v] = llr[v] + sum(r[(c, v)] for c in checks)

            for c in checks:
                q[(v, c)] = llr[v] + sum(
                    r[(d, v)] for d in checks if d != c
                )

        estimate = (posterior < 0).astype(np.int8)

        if not np.any(syndrome(H, estimate)):
            return estimate, True, iteration

    return estimate, False, max_iterations


def simulate(
    H: np.ndarray,
    crossover_probability: float,
    blocks: int = 1000,
    max_iterations: int = 30,
    seed: int = 42,
) -> dict:
    """Monte Carlo evaluation for the all-zero codeword."""
    if blocks <= 0:
        raise ValueError("blocks must be positive.")

    rng = np.random.default_rng(seed)
    n = H.shape[1]
    transmitted = np.zeros(n, dtype=np.int8)

    bp_bit_errors = 0
    bp_block_errors = 0
    bp_success = 0
    bp_iterations = 0

    bf_bit_errors = 0
    bf_block_errors = 0
    bf_success = 0
    bf_iterations = 0

    for _ in range(blocks):
        received, _ = transmit_bsc(transmitted, crossover_probability, rng)

        bp_estimate, bp_ok, bp_it = belief_propagation_decode(
            H, received, crossover_probability, max_iterations
        )
        bf_estimate, bf_ok, bf_it = bit_flip_decode(
            H, received, max_iterations
        )

        bp_errors = int(np.sum(bp_estimate != transmitted))
        bf_errors = int(np.sum(bf_estimate != transmitted))

        bp_bit_errors += bp_errors
        bp_block_errors += int(bp_errors > 0)
        bp_success += int(bp_ok)
        bp_iterations += bp_it

        bf_bit_errors += bf_errors
        bf_block_errors += int(bf_errors > 0)
        bf_success += int(bf_ok)
        bf_iterations += bf_it

    total_bits = blocks * n

    return {
        "crossover_probability": crossover_probability,
        "blocks": blocks,
        "bp_ber": bp_bit_errors / total_bits,
        "bp_block_error_rate": bp_block_errors / blocks,
        "bp_syndrome_success_rate": bp_success / blocks,
        "bp_mean_iterations": bp_iterations / blocks,
        "bitflip_ber": bf_bit_errors / total_bits,
        "bitflip_block_error_rate": bf_block_errors / blocks,
        "bitflip_syndrome_success_rate": bf_success / blocks,
        "bitflip_mean_iterations": bf_iterations / blocks,
    }
