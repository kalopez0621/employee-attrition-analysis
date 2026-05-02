# Employee Attrition Analysis

End-to-end machine learning project analyzing why employees leave IBM, using three techniques (Logistic Regression, K-Means Clustering, Decision Tree) that converged on a single controllable signal: overtime.

**Course:** CAP 4767 Data Mining with Python · Final Capstone Project · Spring 2026

---

## What's in this repo

- **`employee_attrition_analysis.ipynb`** — full analytical workflow: EDA, cleaning, three ML techniques with hyperparameter tuning, model comparison, and business interpretation
- **`business_memo.pdf`** — two-page executive memo addressed to a fictional Chief Human Resources Officer, summarizing findings and recommending three actions
- **`presentation_slides.pdf`** — 10-slide presentation deck used for the capstone defense
- **Video presentation:** https://youtu.be/0JqHjY4zw94

---

## The problem

Employee attrition is expensive. At an industry-estimated replacement cost of $15,000 to $25,000 per employee, a workforce of 1,470 with a 16% annual attrition rate translates to roughly $3.5M to $5.8M in turnover cost per year. The question for HR leadership is not whether attrition is happening, but whether it's predictable, concentrated, and preventable.

This analysis uses the IBM HR Analytics dataset (1,470 employees, 35 features) to identify which employees are most at risk of leaving, what conditions drive their departures, and what management can do about it.

## Approach

Three techniques were applied to the same problem:

1. **Logistic Regression** with cross-validated hyperparameter tuning. Produces interpretable probability scores and ranks feature importance via coefficients. Final model achieved 77% accuracy and 0.80 ROC-AUC.

2. **K-Means Clustering** with K=3 selected via Elbow Method and Silhouette Score. Used to segment employees into behavioral cohorts independent of the attrition label.

3. **Decision Tree** tuned on `max_depth` via cross-validation. Produces plain-English classification rules a manager can act on directly. Final model achieved 73% accuracy and 0.71 ROC-AUC.

Class imbalance (16% positive class) was handled with appropriate strategies during model training.

## Key findings

Three findings emerged consistently across all three techniques:

- **Overtime is the single biggest driver of attrition.** Employees required to work overtime leave at 30.5%, nearly three times the rate of those who do not (10.4%). This was the strongest and most consistent signal in the entire dataset.

- **Risk is concentrated, not company-wide.** Sales Representatives leave at 39.8% per year. Laboratory Technicians at 23.9%. Managers and Research Directors leave at under 5%. Targeted interventions will be far more cost-effective than company-wide programs.

- **One-quarter of the workforce is structurally at risk.** K-Means clustering identified a "Burned Out" segment of 351 employees (24% of workforce) where 100% work overtime and the attrition rate is 34.2%. The cluster framing makes the risk pool actionable rather than diffuse.

## Recommendations

The full memo details three recommended actions in priority order:

1. Audit and reduce mandatory overtime within 30 days
2. Build targeted retention programs for Sales Representatives and Laboratory Technicians
3. Accelerate promotion cycles for early-career employees

## Why three techniques

Triangulation. When three independent methods agree on the same conclusion, the conclusion is more credible than any single model could establish on its own. All three pointed to overtime as the most controllable lever, a finding that holds up across supervised classification, unsupervised segmentation, and rule-based interpretation.

## Tech stack

- **Language:** Python
- **Notebook:** Jupyter (run in Google Colab)
- **Modeling:** scikit-learn (LogisticRegression, KMeans, DecisionTreeClassifier, GridSearchCV)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dataset:** [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)

## Limitations

This analysis is based on a single-company snapshot. It does not include exit interview data, external compensation benchmarks, or employee survey responses. Statistical associations identify *who* is most likely to leave and *under what conditions*, but cannot confirm causal mechanisms.

## About

Built by Karla Lopez. Middle school math teacher (15 years, Miami-Dade County Public Schools), Navy veteran, and dual bachelor's student in Data Analytics + Applied AI at Miami Dade College (Spring 2027).
