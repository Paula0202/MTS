from experiments.run_probabilistic_fidelity import (run_probabilistic_fidelity)

if __name__ == "__main__":
    results = (run_probabilistic_fidelity())
    print(results.head(20))