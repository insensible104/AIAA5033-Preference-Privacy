import argparse
from preference_privacy.rr import generate_rr


def main():
    p = argparse.ArgumentParser(
        description="Prepare reproducible independent RR datasets for paired experiments"
    )
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 1, 2, 4])
    p.add_argument("--rr-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()
    for eps in args.epsilons:
        for seed in args.rr_seeds:
            print(generate_rr(args.train_data, args.output_dir, eps, seed))


if __name__ == "__main__":
    main()
