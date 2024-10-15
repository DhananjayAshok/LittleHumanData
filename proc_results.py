import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib.lines import Line2D

sns.set_theme()
sns.set_style("whitegrid")

def proc_indiviual_results(path="results.csv", title_append=""):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    datasets = df["dataset_name"].unique()
    for dataset in datasets:
        data = df[(df["dataset_name"] == dataset)]
        task = data["task"].iloc[0]
        if "run" in data.columns:
            sns.lineplot(x="frac", y="metric", data=data, hue="run")
        else:
            sns.lineplot(x="frac", y="metric", data=data)
        plt.xlabel("Synthetic Fraction")
        plt.ylabel("Test Accuracy" if task == "Fact Verification" else "BLEU")
        prev_title = title_append
        if dataset.lower() == "scifact":
            for item in ["1000", "3000", "5000"]:
                if item in title_append:
                    title_append = title_append.replace(item, item[:-1])
                    break
        plt.title(f"{dataset} {title_append}")
        plt.savefig(f"figures/{dataset.lower()}.png")
        if dataset.lower() == "scifact" and title_append != prev_title:
            title_append = prev_title
        plt.clf()

def proc_grouped_results(path="results.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    #target_col = "metric"
    datasets = df["dataset_name"].unique()
    for dataset in datasets:
        if "run" not in df.columns:
            dataset_df_indices = df["dataset_name"] == dataset
            base_frac = df[dataset_df_indices & (df["frac"] == 0)]
            if len(base_frac) == 0:
                base_frac = df[dataset_df_indices & (df["frac"] == 0.95)]
            base_value = base_frac["metric"].iloc[0]
            # replace metric with percentage change
            df.loc[dataset_df_indices, "metric"] = (df[dataset_df_indices]["metric"] - base_value) / base_value
        else:
            for run in df["run"].unique():
                dataset_run_df_indices = (df["dataset_name"] == dataset) & (df["run"] == run)
                base_frac = df[dataset_run_df_indices & (df["frac"] == 0)]
                if len(base_frac) == 0:
                    base_frac = df[dataset_run_df_indices & (df["frac"] == 0.95)]
                base_value = base_frac["metric"].iloc[0]
                # replace metric with percentage change
                df.loc[dataset_run_df_indices, "metric"] = (df[dataset_run_df_indices]["metric"] - base_value) / base_value
    df["Dataset"] = df["dataset_name"]
    for task in ["Fact Verification", "Question Answering"]:
        task_df = df[df["task"] == task]
        sns.lineplot(x="frac", y="metric", data=task_df, hue="Dataset", style="Dataset", markers=True, markersize=10)
        # use times new roman
        plt.xlabel("Synthetic Fraction", fontdict={"size": 20, 'fontname':'Times New Roman'})
        if task == "Fact Verification":
            plt.ylabel("% Change in Accuracy", fontdict={"size": 20, 'fontname':'Times New Roman'})
            plt.title(f"Change in Test Accuracy with Increasingly Synthetic Dataset for FV", fontdict={"size": 15, 'fontname':'Times New Roman'})
            plt.show()
            #plt.savefig(f"figures/fv.png")
        else:
            plt.ylabel("% Change in BLEU", fontdict={"size": 20, 'fontname':'Times New Roman'})
            plt.title(f"Change in Test BLEU with Increasingly Synthetic Dataset for QA", fontdict={"size": 15, 'fontname':'Times New Roman'})
            plt.show()
            #plt.savefig(f"figures/qa.png")
        plt.clf()

def proc_money_results(path="results.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    target_cols = None
    target_cols  = list(set(df.columns) - set(["synthetic", "dataset_name", "real"]))
    datasets = df["dataset_name"].unique()
    for dataset in datasets:
        for target_col in target_cols:
            data = df[(df["dataset_name"] == dataset)]
            data["n_points"] = data["synthetic"] + data["real"]
            xrange = (min(data["synthetic"])-200, max(data["synthetic"])+1000)
            synth_data_df = data[(data["real"]) == 0]
            real_data_df = data[(data["real"]) != 0]
            x = np.log(synth_data_df["n_points"].values.reshape(-1, 1))
            y = synth_data_df[target_col].values
            synth_reg = LinearRegression().fit(x, y)
            xrange = np.linspace(xrange[0], xrange[1], 100).reshape(-1, 1)
            synth_yrange = synth_reg.predict(np.log(xrange))
            x = np.log(real_data_df["n_points"].values.reshape(-1, 1))
            y = real_data_df[target_col].values
            real_reg = LinearRegression().fit(x, y)
            real_yrange = real_reg.predict(np.log(xrange))
            #sns.regplot(x="synthetic", y=target_col, data=data_df, color="blue", label="Synthetic Trend")
            sns.scatterplot(x="synthetic", y=target_col, data=synth_data_df, color="blue", alpha=0.3, label="Synthetic")
            sns.lineplot(x=xrange.flatten(), y=synth_yrange, color="blue", label="Synthetic Trend")
            #sns.lineplot(x=xrange.flatten(), y=real_yrange, color="red", label="Real Trend")
            point_df = data[(data["real"] != 0)].reset_index()
            scatter = sns.scatterplot(x="synthetic", y=target_col, data=point_df, color="red", label="w Additional Real")
            points_needed = []
            for i, point in point_df.iterrows():
                n_synth = point['synthetic']
                real_performance = point[target_col]
                base_performance = synth_data_df[synth_data_df['synthetic'] == n_synth].iloc[0][target_col]
                total_n_synth_needed = (real_performance - synth_reg.intercept_) / synth_reg.coef_[0]
                total_n_synth_needed = np.exp(total_n_synth_needed)
                n_synth_needed = total_n_synth_needed - n_synth
                print(f"\tAt {n_synth} Synthetic Points: {n_synth_needed}")
                points_needed.append(n_synth_needed)
                add_plots = [len(point_df)//5, len(point_df)//4, len(point_df)//3]
                if i in add_plots:
                    plt.plot([n_synth, n_synth], [base_performance, real_performance], 'r--')
                    plt.text(min((n_synth_needed+2*n_synth)/2, xrange[-1][0]+1000), real_performance, f'(+{n_synth_needed:.0f})', color='red', ha='center', va='bottom')
                    if i == add_plots[0]:
                        line = plt.plot([n_synth, np.clip(n_synth_needed+n_synth, a_min=0, a_max=xrange[-1][0]+1000)], [real_performance, real_performance], 'r--', label="Synthetic Needed")
                        plt.text(n_synth, (base_performance + real_performance)/2, '(+200)', color='red', ha='center', va='bottom')
                    else:
                        line = plt.plot([n_synth, np.clip(n_synth_needed+n_synth, a_min=0, a_max=xrange[-1][0]+1000)], [real_performance, real_performance], 'r--')
                else:
                    pass
                    #plt.plot([n_synth, n_synth], [base_performance, real_performance], 'r--')
            points_needed = np.array(points_needed)
            points_needed.sort()
            print(f"{dataset} | Mean: {np.mean(points_needed):.2f}, Median: {np.median(points_needed):.2f}, 25th: {points_needed[int(len(points_needed)*0.25):int(len(points_needed)*0.25)+1][0]:.2f}, 75th: {points_needed[int(len(points_needed)*0.75):int(len(points_needed)*0.75)+1][0]:.2f}")
            handles, labels = scatter.get_legend_handles_labels()
            #handles.append(line)
            #labels.append('Synthetic Needed')
            plt.legend(handles=handles, labels=labels)
            plt.xlabel("Number of Synthetic Points")
            plt.ylabel(target_col)
            plt.title(f"{dataset}: {target_col}")
            plt.show()
            #plt.savefig(f"figures/money/{dataset.lower()}_{target_col.lower()}.png")
            #plt.clf()
            print(f"{dataset} Number of Synthetic Points needed to achieve same benefit as 200 real points:")

        


def show_zoom():
    proc_indiviual_results("results.csv", title_append="")
    proc_grouped_results("results.csv")




if __name__ == "__main__":
    show_zoom()
    #proc_money_results()

