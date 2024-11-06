from holisticai.benchmark import BiasMitigationBenchmark
from holisticai.bias.mitigation import CorrelationRemover
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def main():
    benchmark = BiasMitigationBenchmark("binary_classification", "preprocessing")

    sensitive_results = pd.DataFrame()
    for alpha in tqdm(np.linspace(0, 1, 101)):
        my_mitigator = CorrelationRemover(alpha=alpha)
        my_mitigator.__class__.__name__ = f"CorrelationRemover: alpha={alpha:.1f}"
        my_results = benchmark.run(custom_mitigator=my_mitigator)
        sensitive_results = pd.concat([sensitive_results, my_results[f"CorrelationRemover: alpha={alpha:.1f}"]], axis=1)

    if not os.path.exists("_results/sens"):
        os.makedirs("_results/sens")
    sensitive_results.to_csv("_results/sens/alpha_sensitive_analysis_correlation_remover.csv")

    mean_score = sensitive_results[:1].T
    plt.plot(np.linspace(0, 1, 101), mean_score["Mean Score"], label="Mean Score", marker="o", color='purple')
    plt.xlabel("Alpha")
    plt.ylabel("Mean Score")
    plt.legend()
    plt.savefig('_results/sens/binary_classification_preprocessing_sensitivity.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()