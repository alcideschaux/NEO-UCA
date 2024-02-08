import statsmodels.api as sm
import numpy as np
import pandas as pd
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2_contingency, pointbiserialr
import matplotlib.pyplot as plt

# Cramer's V
# x, y: Categorical columns
def cramers_v(x, y):
    # Create a contingency table
    confusion_matrix = pd.crosstab(x, y)
    # Calculate the chi-square statistic
    chi2 = chi2_contingency(confusion_matrix)[0]
    # Calculate the number of observations
    n = confusion_matrix.sum().sum()
    # Calculate the phi-squared statistic
    phi2 = chi2 / n
    # Calculate the number of rows and columns
    r, k = confusion_matrix.shape
    # Calculate the corrected phi-squared statistic
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    # Calculate the corrected row and column counts
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    # Calculate Cramer's V statistic
    cramer_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    cramer_v = round(cramer_v, 2)
    return cramer_v

# Rank-biserial correlation
# x: Categorical column
# y: Numeric column
def rank_biserial_correlation(x, y):
    # Convert the categorical column to numeric codes
    x = pd.Categorical(x).codes
    # Convert the numeric column to numeric values and handle non-numeric values
    y = pd.to_numeric(y, errors='coerce')
    # Create a DataFrame with both columns47
    df = pd.DataFrame({'Categorical':x, 'Numeric':y})
    # Remove rows with 'NaN' values
    df = df.dropna()
    # Calculate the point-biserial correlation
    correlation, p_value = pointbiserialr(df['Categorical'], df['Numeric'])
    correlation = round(correlation, 2)
    return correlation

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

def plot_survival_curves(df, group, time, event):
    """
    Function to plot Kaplan-Meier survival curves for different groups in a dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe containing the survival data.
    group_by (str): The column name to group by.
    time_col (str): The column name for the time to event data.
    event_col (str): The column name for the event indicator (1=event, 0=censored).
    """
    # Group by the specified column
    grouped_df = df.groupby(group)

    # Create a Kaplan-Meier fitter object
    kmf = KaplanMeierFitter()

    # Fit the survival data for each group and plot
    for group, data in grouped_df:
        kmf.fit(data[time], event_observed=data[event])
        kmf.plot(label=group, ci_show=False)
    
    plt.xlabel('Follow-up (months)')
    plt.legend()
    plt.show()

def run_logrank_test(df, group, time, event):
    """
    Runs a log-rank test for comparing the survival distributions of two groups.

    Parameters:
    - dataframe: pandas DataFrame containing the data.
    - time_col: string, name of the column in dataframe that contains the time to event or censoring.
    - group_col: string, name of the column in dataframe that contains the group labels.
    - event_col: string, name of the column in dataframe that contains the event occurrence (1 if event occurred, 0 otherwise).

    Returns:
    - summary of the log-rank test results.
    """
    results = multivariate_logrank_test(df[time], df[group], df[event])
    print(results.summary)

def do_survival(df, group, time, event):
    """
    Plots KM survival curves and runs a log-rank test
    """

    # Plot survival curves
    plot_survival_curves(df, group, time, event)

    # Run logrank test
    run_logrank_test(df, group, time, event)
