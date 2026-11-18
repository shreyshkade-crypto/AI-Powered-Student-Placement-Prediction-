"""
Runway - ML pipeline (3rd-year level)
=====================================
Steps performed
  1. Load + clean the dataset, exploratory data analysis (EDA) plots
  2. Stratified 80/20 train-test split
  3. Classification (placed / not placed): Logistic Regression, Decision Tree,
     Random Forest - each tuned with GridSearchCV (5-fold CV on the TRAIN set only)
  4. Evaluation on the untouched test set: accuracy, precision, recall, F1,
     ROC-AUC, confusion matrix, ROC curves, permutation feature importance
  5. Regression (expected salary of placed students) if the dataset has a `salary` column
  6. Save everything the Flask backend and the research paper need to ml/output/

Usage
-----
    cd ml
    python train_model.py placement_data.csv

Datasets
--------
 * Kaggle "Campus Recruitment" (Placement_Data_Full_Class.csv) - auto-detected.
 * Your own CSV with a binary `placed` column and any of: cgpa, tenth, twelfth,
   aptitude, communication, programming, projects, internships, certifications,
   backlogs (optional `salary` column for the regression part).
"""
import sys, os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
APP_FEATURES = ['cgpa', 'tenth', 'twelfth', 'aptitude', 'communication',
                'programming', 'projects', 'internships', 'certifications', 'backlogs']


# ------------------------------------------------------------------ data
def load_dataset(path):
    df = pd.read_csv(path)
    cols = set(df.columns)
    if {'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'workex', 'status'} <= cols:
        kind = 'kaggle'
        X = pd.DataFrame({
            'ssc_p': df['ssc_p'], 'hsc_p': df['hsc_p'], 'degree_p': df['degree_p'],
            'etest_p': df['etest_p'],
            'workex': (df['workex'].astype(str).str.lower() == 'yes').astype(int)})
        y = (df['status'].astype(str).str.lower() == 'placed').astype(int)
    elif 'placed' in cols:
        kind = 'app'
        feats = [c for c in APP_FEATURES if c in cols]
        if len(feats) < 3:
            sys.exit('Need at least 3 of: ' + ', '.join(APP_FEATURES))
        X, y = df[feats].copy(), df['placed'].astype(int)
    else:
        sys.exit('Unrecognised dataset. See the docstring at the top of this file.')
    salary = pd.to_numeric(df['salary'], errors='coerce') if 'salary' in cols else None
    ok = X.notna().all(axis=1) & y.notna()
    return kind, X[ok].reset_index(drop=True), y[ok].reset_index(drop=True), \
        (salary[ok].reset_index(drop=True) if salary is not None else None)


def eda_plots(X, y):
    """Exploratory data analysis: class balance, feature distributions, correlations."""
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    counts = y.value_counts().sort_index()
    ax.bar(['Not placed', 'Placed'], [counts.get(0, 0), counts.get(1, 0)], color=['0.65', '0.3'])
    ax.set_ylabel('Students'); ax.set_title('Class balance', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'eda_class_balance.png'), dpi=220); plt.close(fig)

    n = X.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.2))
    for a, c in zip(np.atleast_1d(axes), X.columns):
        a.hist(X.loc[y == 0, c], bins=12, alpha=.6, color='0.65', label='Not placed')
        a.hist(X.loc[y == 1, c], bins=12, alpha=.6, color='0.2', label='Placed')
        a.set_title(c, fontsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'eda_distributions.png'), dpi=200); plt.close(fig)

    corr = pd.concat([X, y.rename('placed')], axis=1).corr()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=60, ha='right', fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, fraction=0.046); ax.set_title('Correlation matrix', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'eda_correlation.png'), dpi=220); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    kind, X, y, salary = load_dataset(sys.argv[1])
    n_total = len(X)
    meta_path = sys.argv[1] + '.meta.json'          # written by data/generate_dataset.py
    data_source = json.load(open(meta_path)).get('source', 'real') if os.path.exists(meta_path) else 'real'
    print('Data source:', data_source)
    print(f'Dataset: {kind} | rows: {n_total} | features: {list(X.columns)}')
    print('Class balance (1 = placed):', y.value_counts().to_dict())
    eda_plots(X, y)

    idx_tr, idx_te = train_test_split(np.arange(n_total), test_size=0.20, stratify=y, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    searches = {
        'Logistic Regression': (
            Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=2000, random_state=SEED))]),
            {'m__C': [0.1, 1, 10]}),
        'Decision Tree': (
            DecisionTreeClassifier(random_state=SEED),
            {'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [1, 3, 5]}),
        'Random Forest': (
            RandomForestClassifier(random_state=SEED),
            {'n_estimators': [100, 200], 'max_depth': [4, 6, 8], 'min_samples_leaf': [1, 3]}),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=SEED),
            {'n_estimators': [100, 200], 'max_depth': [2, 3], 'learning_rate': [0.05, 0.1]}),
    }
    rows, fitted, roc_data = [], {}, {}
    for name, (est, grid) in searches.items():
        gs = GridSearchCV(est, grid, cv=cv, scoring='f1', n_jobs=1).fit(X_tr, y_tr)  # train data only
        m = gs.best_estimator_
        pred, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_data[name] = (fpr, tpr)
        rows.append({
            'Model': name,
            'Best params': {k.replace('m__', ''): v for k, v in gs.best_params_.items()},
            'CV F1 (5-fold, train)': round(gs.best_score_, 4),
            'Accuracy': round(accuracy_score(y_te, pred), 4),
            'Precision': round(precision_score(y_te, pred, zero_division=0), 4),
            'Recall': round(recall_score(y_te, pred, zero_division=0), 4),
            'F1': round(f1_score(y_te, pred, zero_division=0), 4),
            'ROC-AUC': round(roc_auc_score(y_te, proba), 4)})
        fitted[name] = (m, pred)
    comp = pd.DataFrame(rows)
    comp.drop(columns='Best params').to_csv(os.path.join(OUT, 'comparison.csv'), index=False)
    print('\n', comp.drop(columns='Best params').to_string(index=False))

    best_name = comp.sort_values('CV F1 (5-fold, train)', ascending=False).iloc[0]['Model']
    best, best_pred = fitted[best_name]
    best_params = comp[comp['Model'] == best_name].iloc[0]['Best params']
    cm = confusion_matrix(y_te, best_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f'\nBest model (chosen by cross-validation on train data): {best_name} {best_params}')
    print('Confusion matrix [[TN FP][FN TP]]:', cm.tolist())

    # confusion matrix
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(cm, cmap='Greys')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center', color='white' if v > cm.max() / 2 else 'black', fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not placed', 'Placed']); ax.set_yticklabels(['Not placed', 'Placed'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(best_name, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'confusion_matrix.png'), dpi=250); plt.close(fig)

    # ROC curves of all models
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} ({comp[comp['Model'] == name].iloc[0]['ROC-AUC']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=.8)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=6.5)
    ax.set_title('ROC curves (test set)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'roc_curves.png'), dpi=250); plt.close(fig)

    # permutation importance (model-agnostic, computed on the test set)
    pi = permutation_importance(best, X_te, y_te, n_repeats=30, random_state=SEED, scoring='f1')
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.barh(np.array(X.columns)[order], pi.importances_mean[order], xerr=pi.importances_std[order], color='0.4')
    ax.set_xlabel('Drop in F1 when feature is shuffled'); ax.set_title('Permutation importance', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=250); plt.close(fig)
    importance = {c: round(float(v), 4) for c, v in zip(X.columns, pi.importances_mean)}

    joblib.dump({'model': best, 'name': best_name, 'features': list(X.columns), 'kind': kind,
                 'medians': X_tr.median().to_dict(), 'data_source': data_source},
                os.path.join(OUT, 'model.joblib'))

    # ---------------- regression: expected salary of placed students
    salary_metrics = None
    if salary is not None and (salary.notna() & (y == 1)).sum() >= 30:
        placed_idx = np.where((y == 1) & salary.notna())[0]
        tr_p = np.array([i for i in placed_idx if i in set(idx_tr)])
        te_p = np.array([i for i in placed_idx if i in set(idx_te)])
        Xs_tr, ys_tr, Xs_te, ys_te = X.iloc[tr_p], salary.iloc[tr_p], X.iloc[te_p], salary.iloc[te_p]
        regs = {'Linear Regression': Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]),
                'Random Forest Regressor': RandomForestRegressor(n_estimators=200, max_depth=5,
                                                                 min_samples_leaf=3, random_state=SEED)}
        reg_rows, reg_fit = [], {}
        for name, r in regs.items():
            cv_r2 = cross_val_score(r, Xs_tr, ys_tr, cv=5, scoring='r2').mean()
            r.fit(Xs_tr, ys_tr)
            p = r.predict(Xs_te)
            reg_rows.append({'Model': name, 'CV R2 (train)': round(float(cv_r2), 4),
                             'MAE': round(float(mean_absolute_error(ys_te, p)), 1),
                             'RMSE': round(float(np.sqrt(mean_squared_error(ys_te, p))), 1),
                             'R2': round(float(r2_score(ys_te, p)), 4)})
            reg_fit[name] = r
        best_reg = max(reg_rows, key=lambda r: r['CV R2 (train)'])['Model']
        joblib.dump({'model': reg_fit[best_reg], 'name': best_reg, 'features': list(X.columns), 'kind': kind},
                    os.path.join(OUT, 'salary_model.joblib'))
        salary_metrics = {'unit': 'INR per year' if kind == 'kaggle' else 'LPA',
                          'n_train': int(len(tr_p)), 'n_test': int(len(te_p)),
                          'best_model': best_reg, 'comparison': reg_rows}
        print('\nSalary regression (placed students only):')
        print(pd.DataFrame(reg_rows).to_string(index=False))

    metrics = {
        'dataset_kind': kind, 'data_source': data_source, 'n_total': int(n_total), 'n_train': int(len(X_tr)), 'n_test': int(len(X_te)),
        'features': list(X.columns), 'best_model': best_name, 'best_params': best_params,
        'best': comp[comp['Model'] == best_name].drop(columns='Best params').iloc[0].to_dict(),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'comparison': comp.drop(columns='Best params').to_dict('records'),
        'permutation_importance': importance, 'salary': salary_metrics}
    json.dump(metrics, open(os.path.join(OUT, 'metrics.json'), 'w'), indent=2, default=str)

    # ---------------- LaTeX macros/tables for the paper (all numbers from this run)
    f = lambda v: f'{v * 100:.1f}'
    esc = lambda t: str(t).replace('_', '\\_')
    tex = [
        f'\\newcommand{{\\mlNTotal}}{{{n_total}}}', f'\\newcommand{{\\mlNTrain}}{{{len(X_tr)}}}',
        f'\\newcommand{{\\mlNTest}}{{{len(X_te)}}}', f'\\newcommand{{\\mlBest}}{{{best_name}}}',
        f'\\newcommand{{\\mlBestParams}}{{{esc(best_params)}}}',
        '\\newcommand{\\mlDataset}{' + ('a synthetic dataset generated for this study' if data_source == 'synthetic' else
                                       'the public Kaggle Campus Recruitment dataset' if kind == 'kaggle'
                                       else 'the institute placement records') + '}',
        '\\newcommand{\\mlFeatures}{' + ', '.join(esc(c) for c in X.columns) + '}',
        f'\\newcommand{{\\mlTN}}{{{tn}}}\\newcommand{{\\mlFP}}{{{fp}}}',
        f'\\newcommand{{\\mlFN}}{{{fn}}}\\newcommand{{\\mlTP}}{{{tp}}}',
        '\\newcommand{\\mlResultsTable}{%', '\\begin{tabular}{@{}lccccc@{}}\\toprule',
        '\\textbf{Model} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Recall} & \\textbf{F1} & \\textbf{AUC} \\\\ \\midrule']
    for r in rows:
        tex.append(f"{r['Model']} & {f(r['Accuracy'])} & {f(r['Precision'])} & {f(r['Recall'])} & {f(r['F1'])} & {r['ROC-AUC']:.2f} \\\\")
    tex.append('\\bottomrule\\end{tabular}}')
    if data_source == 'synthetic':
        tex.append('\\newcommand{\\mlSynthetic}{1}')
    if salary_metrics:
        tex += ['\\newcommand{\\mlHasSalary}{1}',
                f"\\newcommand{{\\mlSalN}}{{{salary_metrics['n_train'] + salary_metrics['n_test']}}}",
                '\\newcommand{\\mlSalaryTable}{%', '\\begin{tabular}{@{}lccc@{}}\\toprule',
                '\\textbf{Model} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{$R^2$} \\\\ \\midrule']
        for r in salary_metrics['comparison']:
            d = 2 if salary_metrics['unit'] == 'LPA' else 0
            tex.append(f"{r['Model']} & {r['MAE']:.{d}f} & {r['RMSE']:.{d}f} & {r['R2']:.2f} \\\\")
        tex.append('\\bottomrule\\end{tabular}}')
    open(os.path.join(OUT, 'ml_results.tex'), 'w').write('\n'.join(tex) + '\n')
    print('\nSaved model + reports in', OUT)


if __name__ == '__main__':
    main()
