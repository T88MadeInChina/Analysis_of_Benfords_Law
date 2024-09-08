import os
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2
from sklearn.linear_model import LinearRegression

def read_csv_file(file_path):
    sep_options = [',', ';', '\t']
    encoding_options = ['utf-8', 'latin1']
    df = None

    for encoding in encoding_options:
        for sep in sep_options:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                return df
            except (pd.errors.ParserError, UnicodeDecodeError):
                continue
    return df


def leading_digit_analysis(data):
    data = data.dropna().astype(str)
    data = data.apply(lambda x: x[1:] if x.startswith('-') else x)
    data = data[data.str.replace('.', '').str.isdigit()]
    leading_digits = data.str[0].astype(int)
    leading_digits = leading_digits[leading_digits != 0]
    leading_digit_counts = leading_digits.value_counts().sort_index()
    total_count = leading_digits.count()
    leading_digit_freq = leading_digit_counts / total_count

    MIN_NONZERO_VALUE = 1e-10
    for i in range(1, 10):
        if i not in leading_digit_freq:
            leading_digit_freq.loc[i] = MIN_NONZERO_VALUE
        elif leading_digit_freq.loc[i] == 0:
            leading_digit_freq.loc[i] = MIN_NONZERO_VALUE

    leading_digit_freq = leading_digit_freq.sort_index()
    return leading_digit_freq

def benford_distribution():
    return np.log10(1 + 1 / np.arange(1, 10))

def chi_square_test(observed, expected, total_count):
    chi2_stat = sum(((observed * total_count - expected * total_count) ** 2) / (expected * total_count))
    return chi2_stat

def z_test(observed, expected, total_count):
    pe = expected
    po = observed
    z_scores = []
    for i in range(len(po)):
        pe_i = pe[i]
        po_i = po[i]
        z_score = (np.sqrt(total_count) * (abs(po_i - pe_i) - (1 / (2 * total_count)))) / np.sqrt(pe_i * (1 - pe_i))
        z_scores.append(z_score)
    return z_scores

def measure_of_distance(observed, expected):
    return np.linalg.norm(observed - expected)

def regression_measure(observed):
    benford_proportions = np.log10(1 + 1 / np.arange(1, 10)).reshape(-1, 1)
    observed_proportions = observed.reshape(-1, 1)
    model = LinearRegression().fit(benford_proportions, observed_proportions)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    return slope, intercept

# Enter the file address and return the analysis results
def analyze_medical_data(file_path, code):
    df = read_csv_file(file_path)
    if df is None or 'ERGEBNISF1' not in df.columns:
        return None

    data = df['ERGEBNISF1']
    digit_freq = leading_digit_analysis(data)
    benford_freq = benford_distribution() * digit_freq.sum()
    total_count = len(data.dropna())

    if len(digit_freq) != 9 or len(benford_freq) != 9:
        print(f"Error: Length mismatch for code = {code}, digit_freq length = {len(digit_freq)}, benford_freq length = {len(benford_freq)}")
        return None

    chi2_stat = chi_square_test(digit_freq, benford_freq, total_count)
    z_scores = z_test(digit_freq.values, benford_freq, total_count)
    distance = measure_of_distance(digit_freq.values, benford_freq)
    slope, intercept = regression_measure(digit_freq.values)

    return [code, chi2_stat, distance, slope, intercept] + z_scores

# Analyze all the files in the folder through a for loop
def main():
    folder_path = 'uploads'
    results = []

    # All files begin with the medical data name and end with roh19.csv (i.e.CODE_roh19.csb)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_roh19.csv'):
            code = file_name.split('_roh19')[0]
            file_path = os.path.join(folder_path, file_name)
            result = analyze_medical_data(file_path, code)
            if result:
                results.append(result)

    columns = ['CODE', 'Chi-Squared', 'SSD', 'Reg_Slope', 'Reg_Intercept'] + [f'Z-Test_{i}' for i in range(1, 10)]
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv('entire_medical_data_results.csv', index=False)

if __name__ == "__main__":
    main()
