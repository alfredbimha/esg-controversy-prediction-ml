# ESG Controversy Prediction (Machine Learning)

## Research Question
Can we predict ESG controversies from firm characteristics?

## Methodology
**Language:** Python  
**Methods:** Logistic Regression, Random Forest, Gradient Boosting, SHAP

## Data
Simulated controversy dataset (1500 firms) calibrated to real distributions

## Key Findings
Logistic regression achieves AUC ~0.74; ESG score, size, and leverage are top predictors.

## How to Run
```bash
pip install -r requirements.txt
python code/project16_*.py
```

## Repository Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── code/          ← Analysis scripts
├── data/          ← Raw and processed data
└── output/
    ├── figures/   ← Charts and visualizations
    └── tables/    ← Summary statistics and regression results
```

## Author
Alfred Bimha

## License
MIT

---
*Part of a 20-project sustainable finance research portfolio.*
