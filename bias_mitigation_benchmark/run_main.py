#!pip install -q git+https://github.com/holistic-ai/holisticai.git@feature/bias-benchmark

import os
from holisticai.benchmark import BiasMitigationBenchmark

def main():
    for task_type in ["multiclass",
                      "regression",
                      "clustering",
                      "binary_classification",
                      ]:
        for stage in ["preprocessing",
                      "inprocessing",
                      "postprocessing"
                      ]:
            benchmark = BiasMitigationBenchmark(task_type, stage)
            results = benchmark.run()
            if not os.path.exists("_results/bias"):
                os.makedirs("_results/bias")
            results.to_csv(f"_results/bias/benchmark_{task_type}_{stage}.csv", index=True)
            print(f"Results saved for {task_type} task type and {stage} stage.")  # noqa: T201

if __name__ == "__main__":
    main()
