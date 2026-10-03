from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Allow execution as: python src/run_experiment.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ldpc import generate_gallager_ldpc, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LDPC BP decoding.")
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--dv", type=int, default=3)
    parser.add_argument("--dc", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    H = generate_gallager_ldpc(
        n=args.n,
        variable_degree=args.dv,
        check_degree=args.dc,
        seed=args.seed,
    )

    # Moderate range for a clear Monte Carlo demonstration.
    probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

    rows = []
    for i, p in enumerate(probabilities):
        print(f"Running p={p:.3f} ...")
        result = simulate(
            H,
            crossover_probability=p,
            blocks=args.blocks,
            max_iterations=args.iterations,
            seed=args.seed + i,
        )
        rows.append(result)

    df = pd.DataFrame(rows)
    csv_path = results_dir / "ldpc_results.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        df["crossover_probability"],
        df["bp_ber"],
        marker="o",
        label="Sum-product BP",
    )
    plt.semilogy(
        df["crossover_probability"],
        df["bitflip_ber"],
        marker="s",
        label="Bit-flipping baseline",
    )
    plt.semilogy(
        df["crossover_probability"],
        df["crossover_probability"],
        linestyle="--",
        label="Raw channel BER",
    )
    plt.xlabel("BSC crossover probability")
    plt.ylabel("Bit error rate (BER)")
    plt.title("LDPC decoding performance")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    figure_path = results_dir / "ber_vs_crossover_probability.png"
    plt.savefig(figure_path, dpi=180)
    plt.close()

    print("\nSaved:")
    print(csv_path)
    print(figure_path)
    print("\nResults:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
