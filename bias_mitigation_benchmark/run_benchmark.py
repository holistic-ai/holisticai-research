from holisticai.benchmark import BiasMitigationBenchmark
import matplotlib.pyplot as plt
import os

def main():
    for task_type in ["binary_classification", "multiclass", "regression", "clustering"]:
            for stage in ["preprocessing", "inprocessing", "postprocessing"]:
                if task_type == "binary_classification" and stage != "preprocessing":
                    continue
                benchmark = BiasMitigationBenchmark(task_type, stage)

                # set file path to save results
                dir_path = os.path.dirname(os.path.realpath(__file__))

                if not os.path.exists(f"{dir_path}/_results"):
                    os.makedirs(f"{dir_path}/_results")
                
                benchmark.get_table().T.to_latex(f'{dir_path}/_results/tab/{task_type}_{stage}.tex', float_format="%.4f")
                
                if not os.path.exists(f"{dir_path}/_results/fig"):
                    os.makedirs(f"{dir_path}/results/fig")
                
                if task_type == "binary_classification":
                    benchmark.get_heatmap(output_path=f'{dir_path}/_results/fig/{task_type}_{stage}.pdf', fig_size=(20, 5))
                else:
                    benchmark.get_heatmap(output_path=f'{dir_path}/_results/fig/{task_type}_{stage}.pdf')
                plt.close()

if __name__ == "__main__":
    main()