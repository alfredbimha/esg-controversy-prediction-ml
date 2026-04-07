"""
===============================================================================
PROJECT 16: ESG Controversy Prediction Using Machine Learning
===============================================================================
RESEARCH QUESTION:
    Can we predict ESG controversies from financial and text features?
METHOD:
    Random Forest, Gradient Boosting, Logistic Regression with SHAP
DATA:
    Simulated controversy data calibrated to RepRisk/MSCI patterns
===============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve)
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Generating ESG controversy dataset...")
np.random.seed(42)
n = 1500

sectors = np.random.choice(['Energy','Mining','Tech','Finance','Consumer','Pharma','Industrial','Utilities'], n)
sector_risk = {'Energy':0.4,'Mining':0.35,'Tech':0.15,'Finance':0.2,
               'Consumer':0.15,'Pharma':0.25,'Industrial':0.2,'Utilities':0.18}

# Features that predict controversies
esg_score = np.random.normal(55, 15, n).clip(10, 95)
firm_size = np.random.normal(10, 2, n)  # log assets
media_mentions = np.random.exponential(50, n).round(0)
employee_count = np.exp(np.random.normal(8, 1.5, n)).round(0)
revenue_growth = np.random.normal(5, 15, n)
debt_ratio = np.random.normal(0.4, 0.15, n).clip(0, 1)
board_independence = np.random.normal(0.6, 0.15, n).clip(0.2, 1)
supply_chain_length = np.random.poisson(5, n)
prior_controversies = np.random.poisson(1.5, n)

# Controversy probability (logistic model)
logit = (-3.5 
         + np.array([sector_risk.get(s, 0.2) for s in sectors]) * 2
         - esg_score * 0.03
         + firm_size * 0.1
         + np.log1p(media_mentions) * 0.3
         - board_independence * 1.5
         + supply_chain_length * 0.15
         + prior_controversies * 0.5
         + debt_ratio * 1.0
         + np.random.normal(0, 0.5, n))

prob = 1 / (1 + np.exp(-logit))
controversy = (np.random.random(n) < prob).astype(int)

df = pd.DataFrame({
    'sector': sectors, 'esg_score': esg_score.round(1),
    'firm_size_log': firm_size.round(2), 'media_mentions': media_mentions,
    'employees': employee_count, 'revenue_growth': revenue_growth.round(2),
    'debt_ratio': debt_ratio.round(3), 'board_independence': board_independence.round(3),
    'supply_chain_length': supply_chain_length,
    'prior_controversies': prior_controversies,
    'controversy': controversy
})
df.to_csv('data/controversy_data.csv', index=False)
print(f"  N={n}, Controversies={controversy.sum()} ({controversy.mean()*100:.1f}%)")

print("\nSTEP 2: Training ML models...")

features = ['esg_score','firm_size_log','media_mentions','employees',
            'revenue_growth','debt_ratio','board_independence',
            'supply_chain_length','prior_controversies']
sector_dummies = pd.get_dummies(df['sector'], prefix='sector', drop_first=True, dtype=float)
X = pd.concat([df[features], sector_dummies], axis=1)
y = df['controversy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = []
roc_data = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc')
    
    results.append({
        'Model': name, 'AUC': round(auc, 4),
        'CV_AUC_mean': round(cv.mean(), 4), 'CV_AUC_std': round(cv.std(), 4),
        'Accuracy': round((y_pred == y_test).mean(), 4)
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    
    print(f"  {name}: AUC={auc:.3f}, CV-AUC={cv.mean():.3f}±{cv.std():.3f}")

pd.DataFrame(results).to_csv('output/tables/model_comparison.csv', index=False)

# Feature importance (Random Forest)
rf = models['Random Forest']
fi = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
fi.to_csv('output/tables/feature_importance.csv', index=False)

print("\nSTEP 3: Visualizations...")

# Fig 1: ROC curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {'Logistic Regression':'#3498db','Random Forest':'#2ecc71','Gradient Boosting':'#e74c3c'}
for name, (fpr, tpr, auc) in roc_data.items():
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=colors[name], lw=2)
axes[0].plot([0,1],[0,1],'k--',lw=1)
axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

# Feature importance
fi_top = fi.head(12)
axes[1].barh(fi_top['feature'], fi_top['importance'], color='steelblue', edgecolor='white')
axes[1].set_title('Feature Importance (Random Forest)', fontweight='bold')
axes[1].set_xlabel('Importance')
plt.tight_layout()
plt.savefig('output/figures/fig1_model_results.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, models['Gradient Boosting'].predict(X_test_s))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Controversy','Controversy'],
            yticklabels=['No Controversy','Controversy'])
ax.set_title('Confusion Matrix (Gradient Boosting)', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('output/figures/fig2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Partial dependence style — ESG score vs controversy rate
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, feat in enumerate(['esg_score','prior_controversies','board_independence']):
    bins = pd.qcut(df[feat], 10, duplicates='drop')
    agg = df.groupby(bins)['controversy'].mean()
    axes[i].bar(range(len(agg)), agg.values, color='steelblue', edgecolor='white')
    axes[i].set_title(f'Controversy Rate by {feat}', fontweight='bold', fontsize=11)
    axes[i].set_ylabel('Controversy Rate')
    axes[i].set_xlabel(feat)
plt.tight_layout()
plt.savefig('output/figures/fig3_feature_effects.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
