import statsmodels.api as sm
import numpy as np
import pandas as pd
import lifelines

def logistic_regression(data, X, y):
    # Define the dependent variable
    y = data[y]

    # Define the independent variable (don't forget to add a constant)
    X = sm.add_constant(data[X])

    # Create the logistic regression model
    logreg = sm.Logit(y, X)

    # Fit the model to the data
    result = logreg.fit()

    # Obtain the odds ratio, confidence interval, and p-values
    odds_ratios = np.exp(result.params)  # Exponentiate the model coefficients
    conf_int = np.exp(result.conf_int())  # Exponentiate the confidence intervals
    p_values = result.pvalues  # Get the p-values

    # Create a DataFrame to display the results
    odds_ratios = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'Lower CI': conf_int[0],
        'Upper CI': conf_int[1],
        'P-value': p_values
    }).iloc[1:]

    return odds_ratios

def cox_regression(data, duration, formula, event):
    # Fit the Cox proportional hazards model
    cph = lifelines.CoxPHFitter()
    cph.fit(data, duration_col=duration, event_col=event, formula=formula)

    renamed_summary = cph.summary.rename(columns={'exp(coef)': 'Hazards Ratio', 'p': 'P-value', 'exp(coef) lower 95%': 'Lower CI', 'exp(coef) upper 95%': 'Upper CI'})

    return renamed_summary[['Hazards Ratio','Lower CI','Upper CI','P-value']]