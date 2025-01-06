import numpy as np
import pandas as pd
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter
from scipy.stats import chi2_contingency, pointbiserialr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit

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

def logistic_analysis(data, outcome, predictors, print_full_summary=True):
    """
    Perform logistic regression analysis for specified outcome and predictors
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing all variables
    outcome : str
        Name of the outcome variable (must be binary)
    predictors : list
        List of predictor variables
    print_full_summary : bool, optional
        Whether to print the full regression summary (default False)
    
    Returns:
    --------
    DataFrame with odds ratios, CIs, and p-values
    """

    # Convert outcome to binary if it's 'Yes/No'
    if data[outcome].dtype == 'object':
        data[outcome] = (data[outcome] == 'Yes').astype(int)
    
    # Prepare predictors for formula
    formula_predictors = []
    for pred in predictors:
        # Check if variable is categorical
        if data[pred].dtype == 'object' or data[pred].dtype.name == 'category':
            formula_predictors.append(f"C({pred})")
            data[pred] = data[pred].astype('category')
        else:
            formula_predictors.append(pred)
            data[pred] = data[pred].astype(float)
    
    # Create formula
    formula = f"{outcome} ~ " + " + ".join(formula_predictors)
    
    # Fit model
    model = logit(formula, data=data).fit()
    
    # Calculate odds ratios and CIs
    odds_ratios = np.exp(model.params)
    conf_ints = np.exp(model.conf_int())
    p_values = model.pvalues
    
    # Create results dataframe
    results = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'CI Lower': conf_ints[0],
        'CI Upper': conf_ints[1],
        'P-value': p_values
    })
    
    # Print results
    print(f"\nLogistic Regression Analysis for {outcome}")
    print("-" * 50)
    for index, row in results.iterrows():
        print(f"{index}:")
        print(f"OR: {row['Odds Ratio']:.3f}, 95% CI: ({row['CI Lower']:.3f}, {row['CI Upper']:.3f}), P={row['P-value']:.3f}")
        print()
    
    if print_full_summary:
        print("\nFull Model Summary:")
        print(model.summary())
        print("\nModel Fit Statistics:")
        print(f"AIC: {model.aic:.2f}")
        print(f"BIC: {model.bic:.2f}")
        print(f"Pseudo R-squared: {model.prsquared:.3f}")
    
    return results

# Example usage:

# For univariate analysis
# logistic_analysis(data, 'recurrence', ['age'])
# logistic_analysis(data, 'recurrence', ['sex'])
# logistic_analysis(data, 'recurrence', ['variant_histology'])

# For multivariate analysis
# logistic_analysis(data, 'recurrence', ['age', 'sex', 'variant_histology', 'ypT_group'])

# For other outcomes
# logistic_analysis(data, 'dod', ['age', 'sex', 'variant_histology', 'ypT_group'])
# logistic_analysis(data, 'dre', ['age', 'sex', 'variant_histology', 'ypT_group'])

def cox_analysis(data, time_var, event_var, predictors, print_full_summary=False):
    """
    Perform Cox proportional hazards analysis
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    import pandas as pd
    import numpy as np
    from lifelines import CoxPHFitter
    
    # Create copy of data with only necessary columns
    df = data[[time_var, event_var] + predictors].copy()
    
    # Convert event to binary if it's 'Yes/No'
    if df[event_var].dtype == 'object':
        df[event_var] = (df[event_var] == 'Yes').astype(int)
    
    # Handle categorical variables with dummy coding
    for pred in predictors:
        if df[pred].dtype == 'object' or df[pred].dtype.name == 'category':
            # Create dummy variables, drop_first=True for reference category
            dummies = pd.get_dummies(df[pred], prefix=pred, drop_first=True)
            # Add dummy columns to dataframe
            df = pd.concat([df, dummies], axis=1)
            # Drop original categorical column
            df = df.drop(columns=[pred])
    
    # Initialize and fit Cox model
    cph = CoxPHFitter()
    cph.fit(df, duration_col=time_var, event_col=event_var)
    
    # Get summary as DataFrame
    summary_df = cph.summary
    
    # Extract results
    results = pd.DataFrame({
        'Hazard Ratio': summary_df['exp(coef)'],
        'CI Lower': summary_df['exp(coef) lower 95%'],
        'CI Upper': summary_df['exp(coef) upper 95%'],
        'P-value': summary_df['p']
    })
    
    # Print results
    print(f"\nCox Proportional Hazards Analysis")
    print(f"Time variable: {time_var}")
    print(f"Event variable: {event_var}")
    print("-" * 50)
    
    for index, row in results.iterrows():
        print(f"{index}:")
        print(f"HR: {row['Hazard Ratio']:.3f}, 95% CI: ({row['CI Lower']:.3f}, {row['CI Upper']:.3f}), P={row['P-value']:.3f}")
        print()
    
    if print_full_summary:
        print("\nFull Model Summary:")
        print(cph.print_summary())
        
        # Print model fit statistics
        print("\nModel Fit Statistics:")
        print(f"Log-likelihood ratio test: {cph.log_likelihood_ratio_test()}")
        print(f"Concordance Index: {cph.concordance_index_:.3f}")
        
        # Check proportional hazards assumption
        print("\nProportional Hazards Test:")
        try:
            assumptions_test = cph.check_assumptions(df)
            print(assumptions_test)
        except Exception as e:
            print("Could not perform proportional hazards test:")
            print(str(e))
    
    return results

# Example usage:
# Read data
# data = pd.read_csv('DATA.csv')

# Univariate analysis
# result1 = cox_analysis(data, time_var='fu_censor', event_var='dod', predictors=['age'])

# Multivariate analysis
# result2 = cox_analysis(data, time_var='fu_censor', event_var='dod', predictors=['age', 'sex', 'variant_histology', 'ypT_group'], print_full_summary=True)

# Analysis for different outcome
# result3 = cox_analysis(data, time_var='fu_recurrence', event_var='recurrence', predictors=['age', 'sex', 'variant_histology', 'ypT_group'])

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
    
    plt.xlabel('Follow-up (Months)')
    plt.ylabel('Survival Probability')
    plt.ylim(0.0, 1.0)
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
