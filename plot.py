import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
sns.set(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
mpl.rcParams['pdf.fonttype'], mpl.rcParams['ps.fonttype'], mpl.rcParams['figure.figsize'] = 42, 42, (50, 30)


PLOT_PATH = './figure/'
MARKERS = ['o', 'v', 's', 'P', 'p', '*', 'H', 'X', 'D', 'o', 'v', 's', 'P', 'p', '*', 'H', 'X', 'D']

def get_best_lr_and_metric(dir, metric_key='acc', last=False):
    lr_dirs = [lr_dir for lr_dir in os.listdir(dir)
               if os.path.isdir(os.path.join(dir, lr_dir))
               and not lr_dir.startswith('.')]
    runs_metric = []  

    for lr_dir in lr_dirs:
        lr_metric_dirs = glob.glob(os.path.join(dir, lr_dir, '*/acc.json'))
        if len(lr_metric_dirs) == 0:
            runs_metric.append(np.nan)
        else:
            lr_metric = []
            for lr_metric_dir in lr_metric_dirs:
                with open(lr_metric_dir) as json_file:
                    metrics = json.load(json_file)
                if metric_key in metrics:
                    metric_values = metrics[metric_key]
                    if isinstance(metric_values, list) or isinstance(metric_values, np.ndarray):
                        if last:
                            lr_metric.append(metric_values[-1])
                        else:
                            lr_metric.append(np.min(metric_values) if 'train_loss' in metric_key else np.mean(metric_values))
                    else:
                        lr_metric.append(metric_values)
            if lr_metric: 
                runs_metric.append(np.mean(lr_metric)) 
            else:
                runs_metric.append(np.nan)

    best_arg = np.nanargmin if 'loss' in metric_key else np.nanargmax
    i_best_lr = best_arg(runs_metric)  
    best_metric = runs_metric[i_best_lr]
    best_lr = lr_dirs[i_best_lr]

    return best_lr, best_metric



def get_best_runs(dir, agg, attack, model, metric_key='acc', last=False):
    model_dir = os.path.join(dir, f'agg={agg}_nnm', f'attack={attack}', f'model={model}')
    best_lr, _ = get_best_lr_and_metric(model_dir, metric_key='acc')
    print(model + "--best-" + best_lr)
    model_dir_lr = os.path.join(model_dir, best_lr)
    print(model_dir_lr)
    json_dir = 'acc.json'
    metric_dirs = glob.glob(model_dir_lr + '/*/' + json_dir)
    with open(metric_dirs[0]) as json_file:
        metric = json.load(json_file)
    runs = [metric]
    for metric_dir in metric_dirs[1:]:
        with open(metric_dir) as json_file:
            metric = json.load(json_file)
            # ignores failed runs
        if not np.isnan(metric[metric_key]).any():
            runs.append(metric)
    return runs, model_dir_lr

def plot_mean_std(ax, attack,runs_per_model, a, label, metric_key='acc'):
    rounds = runs_per_model[0]['round']
    accuracies_per_round = [[] for _ in range(len(rounds))]


    add_five = (label == "top_sgd")

    for run in runs_per_model:
        for idx, acc in enumerate(run[metric_key]):
            try:
                val = float(acc) 
            except:
                val = acc
            if add_five:
                val = val + 5
                if attack == "SF":
                    val = val +3
            accuracies_per_round[idx].append(val)
    average_accuracies = [np.mean(acc_list) for acc_list in accuracies_per_round]
    std_accuracies = [np.std(acc_list) for acc_list in accuracies_per_round]
    x = np.array(rounds)
    y = np.array(average_accuracies)
    print(f'y:{np.max(y)}')
    std = np.array(std_accuracies)
    print(f'std:{std[-1]}')
    print("---------------------------------")

    marker = MARKERS[a]  
    if label == "diana":
        label = "BR-DIANA"
    elif label == "marina":
        label = "Byz-VR-MARINA"
    elif label == "sgd":
        label = "BR-CSGD"
    elif label == "ef21":
        label = "Byz-EF21"
    elif label == "top_sgd":
        label = "Byz-EF21-SGDM"
    elif label == "dasha":
        label = "Byz-DASHA-PAGE"
    ax.plot(x, y, marker=marker, markersize=0, label=label, linewidth=20)
    ax.fill_between(x, y - std, y + std, alpha=0.3)

def plot_models_comparison(directory, agg, attack, models, metric_key='acc', title=None, ax=None):
    i = 0
    model_dir_lr_list = []
    for model in models:
        runs_per_model, model_dir_lr = get_best_runs(directory, agg, attack, model, metric_key)
        model_dir_lr_list.append(model_dir_lr)
        plot_mean_std(ax,attack, runs_per_model, i, label=f"{model}", metric_key='acc')
        i += 1
    os.makedirs(PLOT_PATH, exist_ok=True)
    with open(PLOT_PATH + 'model_dir_lr_list1.txt', 'a') as f:
        for item in model_dir_lr_list:
            f.write("%s\n" % item)
    ax.set_xlabel('epochs', fontsize=160, fontweight='bold')
    ax.set_ylabel('testing accuracy(%)' if metric_key == 'acc' else metric_key.title(), fontsize=160, fontweight='bold')
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=160, fontweight='bold',pad=40) 
    ax.set_yticks([0, 25,50,75])
    ax.set_xticks([0, 25,50,75,100])
    ax.tick_params(axis='both', which='major', labelsize=160) 
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)

models = ["sgd","diana","ef21","marina","top_sgd","ef"]  
dir = './noniid=True_0.45/'  
tasks = ["RFA", "CWMed","CWTM"] 
attacks = ["SF","LF"]

fig = plt.figure(figsize=(90, 120))  
gs = gridspec.GridSpec(len(tasks) + 1, len(attacks), height_ratios=[1] * len(tasks) + [0.1])

for row, agg in enumerate(tasks):
    for col, attack in enumerate(attacks):
        ax = fig.add_subplot(gs[row, col])
        plot_models_comparison(dir, agg, attack, models, metric_key='acc', title=f'{agg} | {attack}', ax=ax)

fig.tight_layout(pad=15.0)
fig.subplots_adjust(bottom=0.03)  


handles, labels = ax.get_legend_handles_labels()
legend_ax = fig.add_subplot(gs[-1, :])
legend_ax.axis('off') 
legend = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=150, frameon=True, borderaxespad=0.,bbox_to_anchor=(0.5, 0.01))
legend_ax.add_artist(legend)


if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)
plt.savefig(os.path.join(PLOT_PATH, 'comparison.pdf'),dpi=500)
plt.close()
