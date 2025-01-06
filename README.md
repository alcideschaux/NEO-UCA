# Differential Outcomes in Bladder Cancer After Neoadjuvant Chemotherapy: Comparing Isolated Nodal Disease versus Persistent Muscle-Invasive Disease

## Background
The prognostic significance of residual lymph node disease after neoadjuvant chemotherapy (NAC) in bladder cancer remains poorly understood. This multi-institutional study evaluated outcomes based on patterns of residual disease.

## Methods
We retrospectively analyzed 174 patients who underwent radical cystectomy following NAC between 2010 and 2023 at academic centers from the United States and Europe. Patients were stratified into two groups: those with isolated lymph node disease despite complete local response (ypT0/Tis/Ta, n=35) and those with persistent muscle-invasive disease without lymph node involvement (ypT2/3, n=139). Primary outcomes included recurrence, disease-specific mortality, and survival. Median follow-up was 27.0 months (interquartile range: 9.0-60.0).

## Data Analysis
1. [Descriptive:](01-EDA.ipynb) Description of the variables of the dataset
2. [Stationarity:](02-STATIONARITY.ipynb) Evaluation of the effect of year of cystectomy on endpoints
3. [Associative:](03-ASSOCIATION.ipynb) Evaluation of the association between clinicopathologic variables and the endpoints
4. [Risks:](04-RISKS.ipynb) Estimation of odds ratios (OR) and hazard ratios (HR) for the endpoints
5. [Survival:](05-SURVIVAL.ipynb) Survival curves and logrank tests for the endpoints

## Results
Disease recurrence occurred in 33% of patients, with significantly higher risk in the isolated lymph node group compared to those with persistent muscle-invasive disease (adjusted OR: 0.43, 95% CI: 0.20-0.95, P=0.036). Disease-specific mortality was 24%, with no significant difference between groups (adjusted OR: 0.70, 95% CI: 0.29-1.64, P=0.407). Disease-related events occurred in 41% of patients, with lower risk in the muscle-invasive group (adjusted OR: 0.46, 95% CI: 0.21-0.99, P=0.048). Survival analyses showed no significant differences in overall or disease-specific survival between groups (HR: 1.03, 95% CI: 0.48-2.20, P=0.947). Variant histology (present in 36% of cases) did not significantly influence outcomes.

## Conclusions
Patients with isolated lymph node disease despite complete local response after NAC demonstrate higher recurrence risk compared to those with persistent muscle-invasive disease, although this does not translate into survival differences. These findings suggest the need for risk-adapted surveillance strategies and consideration of additional therapeutic interventions in patients with isolated residual nodal disease.