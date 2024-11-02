import pandas as pd
import seaborn as sns 
from constants import T_BINARY
import matplotlib.pyplot as plt 
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score


def plot_categorical_distribution(df, column, palette='Set2', stat="count", hue=None, figsize=(6,5)):
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=column, palette=palette, stat=stat, hue=hue)
    plt.title(f'{column} Distribution')
    plt.show()


def calculate_distribution(df, row_column, col_column):
    crosstab = pd.crosstab(df[row_column], df[col_column], normalize='columns') * 100
    crosstab = crosstab.round(2)
    return crosstab


def plot_propensity_hist(propensity_df, model):   
    plt.hist(propensity_df[propensity_df[T_BINARY] == 1]['propensity_score'], fc=(0, 0, 1, 0.5), bins=20, label='Treated')
    plt.hist(propensity_df[propensity_df[T_BINARY] == 0]['propensity_score'], fc=(1, 0, 0, 0.5), bins=20, label='Control')
    plt.legend()
    plt.title(f'{model} propensity scores overlap')
    plt.yscale('log')
    plt.xlabel('propensity score')
    plt.ylabel('number of units (log scaled)')
    plt.show()


def plot_propensity_models_hist(propensities_dict):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for i, (model, propensity_df) in enumerate(propensities_dict.items()):
        axs[i].hist(propensity_df[propensity_df[T_BINARY] == 1]['propensity_score'], 
                    fc=(0, 0, 1, 0.5), bins=20, label='Treated')
        axs[i].hist(propensity_df[propensity_df[T_BINARY] == 0]['propensity_score'], 
                    fc=(1, 0, 0, 0.5), bins=20, label='Control')
        axs[i].set_title(f'{model} propensity scores overlap')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('propensity score')
        axs[i].set_ylabel('number of units (log scaled)')
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def plot_calibration_curve(models_dict, t_true):
    plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for model_name, propensity_df in models_dict.items():
        propensity = propensity_df['propensity_score']
        fraction_of_positives, mean_predicted_value = calibration_curve(t_true, propensity, n_bins=10)

        ax1.plot(mean_predicted_value,
                 fraction_of_positives,
                 "s-",
                 label=model_name)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration curves')
    ax1.set_xlabel("Mean predicted value")
    plt.tight_layout()
    plt.show()


def plot_evaluation_metrics(models_dict, t_true):
    scores = []
    for model_name, propensity_df in models_dict.items():
        propensity = propensity_df['propensity_score']
        
        # Calculate the metrics
        logloss = log_loss(t_true, propensity)
        roc_auc = roc_auc_score(t_true, propensity)
        pr_auc = average_precision_score(t_true, propensity)
        brier = brier_score_loss(t_true, propensity)
        
        # Store the results
        scores.append((model_name, 'Log Loss', logloss))
        scores.append((model_name, 'ROC AUC', roc_auc))
        scores.append((model_name, 'PR AUC', pr_auc))
        scores.append((model_name, 'Brier', brier))
    
    # Convert to a DataFrame for easier plotting
    scores_df = pd.DataFrame(scores, columns=['Model', 'Metric', 'Score'])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Score', hue='Model', data=scores_df)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Model')
    
    # Add labels above the bars
    for p in ax.patches[:-4]:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), textcoords = 'offset points')
    plt.show()


def plot_learners_evaluation_metrics(models_eval_dict):
    scores = []
    for model_name, eval_dict in models_eval_dict.items():
        scores.append((model_name, 'Accuracy', eval_dict['accuracy']))
        scores.append((model_name, 'F1', eval_dict['f1']))
    
    # Convert to a DataFrame for easier plotting
    scores_df = pd.DataFrame(scores, columns=['Model', 'Metric', 'Score'])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Score', hue='Model', data=scores_df)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Model')
    
    # Add labels above the bars
    for p in ax.patches[:-3]:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), textcoords = 'offset points')
    plt.show()