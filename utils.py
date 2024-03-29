import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import bootstrap


def sensitivity_wrapper(true, pred):
    return metrics.recall_score(true, pred)


def specifity_wrapper(true, pred):
    return metrics.recall_score(true, pred, pos_label=0)


def AUC_wrapper(true, pred):
    return metrics.roc_auc_score(true, pred)

def f1_wrapper(true, pred):
    return metrics.f1_score(true, pred)


def get_results(true, pred, proba, text=''):
    res = bootstrap((true, proba), AUC_wrapper, vectorized=False,
                    paired=True, random_state=2401).confidence_interval
    space = ' '*(30 - len(text) - len('ROC AUC'))
    print(f'{text} ROC AUC {space}'
          f'{round(AUC_wrapper(true, proba),2):<5} '
          f'[{round(res[0], 2)}-{round(res[1], 2)}]')
    test_list = [sensitivity_wrapper, specifity_wrapper, f1_wrapper]
    test_names = ['Sensitivity:', 'Specifity:', 'F1 Score:']
    for num, test in enumerate(test_list):
        res = bootstrap((true, pred), test, vectorized=False,
                        paired=True, random_state=2401).confidence_interval
        space = ' '*(30 - len(text) - len(test_names[num]))
        print(f'{text} {test_names[num]} {space}'
              f'{round(test(true, pred),2):<5} '
              f'[{round(res[0], 2)}-{round(res[1], 2)}]')
    print('')
    return None


def get_roc_curve(model):
    plt.clf()
    fpr0, tpr0, _ = metrics.roc_curve(
        model.y_train, model.predict(model.X_train)[1])
    auc0 = metrics.roc_auc_score(
        model.y_train, model.predict(model.X_train)[1])
    fpr1, tpr1, _ = metrics.roc_curve(
        model.y_test, model.predict(model.X_test)[1])
    auc1 = metrics.roc_auc_score(model.y_test, model.predict(model.X_test)[1])
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=False)
    data = {'Training': (ax0, fpr0, tpr0, auc0),
            'Test': (ax1, fpr1, tpr1, auc1)}
    fig.set_size_inches(10, 5)
    for key, value in data.items():
        ax, fpr, tpr, auc = value
        ax.plot(fpr, tpr, color="#007ee2",
                     label=f"ROC, AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], color="#00bac4",
                     linestyle="--", label="Guessing")
        ax.set_xlabel("False positive rate", fontsize=12)
        ax.set_ylabel("True positive rate", fontsize=12)
        ax.legend(loc=4, fontsize=12)
        ax.set_title(key, fontsize=16)
    fig.savefig('svm_roc_auc.png', transparent=False,
                dpi=400, bbox_inches="tight")
    return None


def get_boxplot(model):
    plt.clf()
    individual_feature_importance = np.multiply(
        model.svm.estimator.coef_[0], model.X_train)
    max_value = np.max(np.abs(individual_feature_importance.values))
    df = pd.DataFrame(individual_feature_importance,
                      columns=model.X_train.columns)
    df.boxplot(
        showfliers=True,
        vert=False,
        figsize=(9, 6),
        boxprops={'linestyle': '-', 'color': '#007ee2', 'linewidth': 1},
        whiskerprops={'linestyle': '--', 'color': '#007ee2', 'linewidth': 1},
        medianprops={'linestyle': '-', 'color': '#00bac4', 'linewidth': 1},
        flierprops={'marker': '.',
                    'markeredgecolor': 'black', 'markersize': 3},
    )
    plt.axvline(x=0, color='black', ls=':', linewidth=1)
    plt.gca().invert_yaxis()
    plt.grid(visible=False, axis='y')
    plt.xlim((-1.1 * max_value, 1.1 * max_value))
    plt.xlabel('Individual Feature Importance [a.u.]', fontsize=12, labelpad=5)
    plt.savefig('boxplots.png', transparent=False,
                dpi=400, bbox_inches="tight")
    return None


def get_table_2(transformer, model):
    feature, impute, mean, std = [], [], [], []
    for pipeline in transformer.transformers_:
        feature += pipeline[2]
        impute += pipeline[1].steps[0][1].statistics_.tolist()
        try:
            mean += pipeline[1].steps[1][1].mean_.tolist()
            std += pipeline[1].steps[1][1].scale_.tolist()
        except:
            mean += [np.nan for _ in range(len(pipeline[2]))]
            std += [np.nan for _ in range(len(pipeline[2]))]
    df = pd.DataFrame({
        'feature': feature,
        'default': np.round(impute, 2),
        'mean':  np.round(mean, 2),
        'std': np.round(std, 2),
        'coefficient': model.svm.estimator.coef_[0]
    })
    df.replace(to_replace=np.nan, value='-', inplace=True)
    df.sort_values(['coefficient'], ascending=False, inplace=True)
    print(df.to_string(index=False), '\n')
    return None


def get_intercept(model):
    print('Intercept: ', model.svm.estimator.intercept_[0])
    return None


def get_platt_scaling(model):
    params = np.array([
        model.svm.calibrated_classifiers_[0].calibrators[0].a_,
        model.svm.calibrated_classifiers_[0].calibrators[0].b_])
    print(f'Platt Scaling: A = {params[0]}, B = {params[1]}')
    return None


if __name__ == "__main__":
    print("Import this file as a module.")