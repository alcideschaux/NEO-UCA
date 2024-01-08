from scipy.stats import chi2_contingency, pointbiserialr
import numpy as np
import pandas as pd

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

    # Create a DataFrame with both columns
    df = pd.DataFrame({'Categorical':x, 'Numeric':y})

    # Remove rows with 'NaN' values
    df = df.dropna()

    # Calculate the point-biserial correlation
    correlation, p_value = pointbiserialr(df['Categorical'], df['Numeric'])
    correlation = round(correlation, 2)

    return correlation
