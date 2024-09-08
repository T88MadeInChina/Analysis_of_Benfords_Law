import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def generate_distribution_data(ul_values, step=1):
    # Generate normally distributed data sets
    def ul_rnorm(ul):
        mu = (1 + ul / 2)
        sigma = max((ul - 1) / (norm.ppf(0.975) - norm.ppf(0.025)), 1e-10)
        return np.random.normal(loc=mu, scale=sigma, size=100000)

    # Generate lognormally distributed datasets
    def ul_rlnorm(ul):
        lul = np.log(ul)
        mu_log = (1 + lul) / 2
        sigma_log = max((lul - 1) / (norm.ppf(0.975) - norm.ppf(0.025)), 1e-10)
        return np.random.lognormal(mean=mu_log, sigma=sigma_log, size=100000)

    norm_data = {}
    lognorm_data = {}


    # The upper limit of the normally and lognormally
    # distributed dataset is increasing, and the upper limit
    # is maximized by ul_values
    for ul in range(2, ul_values + 1, step):
        norm_data[ul] = ul_rnorm(ul)
        lognorm_data[ul] = ul_rlnorm(ul)


    # Save the dataset in a csv file and represent it by the value of ul
    norm_df = pd.DataFrame.from_dict(norm_data, orient='index').transpose()
    lognorm_df = pd.DataFrame.from_dict(lognorm_data, orient='index').transpose()

    norm_df.to_csv(f'norm_data_{ul_values}.csv', index=False)
    lognorm_df.to_csv(f'lognorm_data_{ul_values}.csv', index=False)

    return norm_df, lognorm_df



def analyze_distribution_data(norm_df, lognorm_df, ul_values):
    def leading_digit_analysis(data):
        data = data.dropna().astype(str)
        # Excluding negative numbers
        data = data.apply(lambda x: x[1:] if x.startswith('-') else x)
        data = data[data.str.replace('.', '').str.isdigit()]
        leading_digits = data.str[0].astype(int)
        # Excluding cases where the first digit is 0
        leading_digits = leading_digits[leading_digits != 0]
        leading_digit_counts = leading_digits.value_counts().sort_index()
        total_count = leading_digits.count()
        leading_digit_freq = leading_digit_counts / total_count

        # Set the minimum value of the frequency so that it is not 0
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
            z_score = ((np.sqrt(total_count) * (abs(po_i - pe_i) -(1 / (2 * total_count)))) /
                       np.sqrt(pe_i * (1 - pe_i)))
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

    def read_and_process_df(df):
        results = []

        for ul in df.columns:
            data = pd.Series(df[ul])
            digit_freq = leading_digit_analysis(data)
            benford_freq = benford_distribution() * digit_freq.sum()
            total_count = len(data.dropna())

            if len(digit_freq) != 9 or len(benford_freq) != 9:
                print(f"Error: Length mismatch for ul = {ul}, digit_freq length = {len(digit_freq)}, benford_freq length = {len(benford_freq)}")
                continue

            chi2_stat = chi_square_test(digit_freq, benford_freq, total_count)
            z_scores = z_test(digit_freq.values, benford_freq, total_count)
            distance = measure_of_distance(digit_freq.values, benford_freq)
            slope, intercept = regression_measure(digit_freq.values)

            results.append([ul, chi2_stat, z_scores, distance, slope, intercept])

        return results

    def save_results_to_csv(results, file_path):
        columns = ['ul', 'Chi-Squared', 'SSD', 'Reg_Slope', 'Reg_Intercept'] + [f'Z-Test_{i}' for i in range(1, 10)]
        processed_results = []
        for result in results:
            ul, chi2_stat, z_scores, distance, slope, intercept = result
            processed_results.append([ul, chi2_stat, distance, slope, intercept] + z_scores)
        results_df = pd.DataFrame(processed_results, columns=columns)
        results_df.to_csv(file_path, index=False)


    # Save the results to a csv file and also represent the results of
    # the analysis of this dataset by the value of ul
    norm_results = read_and_process_df(norm_df)
    lognorm_results = read_and_process_df(lognorm_df)

    save_results_to_csv(norm_results, f'norm_results_{ul_values}.csv')
    save_results_to_csv(lognorm_results, f'lognorm_results_{ul_values}.csv')

    return norm_results, lognorm_results



# Visualize the results
def visualize_data(norm_results_csv, lognorm_results_csv):
    norm_results = pd.read_csv(norm_results_csv)
    lognorm_results = pd.read_csv(lognorm_results_csv)

    # Calculate p-value for chi-square test
    norm_results['Chi-Squared_p_value'] = 1 - chi2.cdf(norm_results['Chi-Squared'], df=8)
    lognorm_results['Chi-Squared_p_value'] = 1 - chi2.cdf(lognorm_results['Chi-Squared'], df=8)

    # Calculate the p-value for the Z-test
    z_columns = [f'Z-Test_{i}' for i in range(1, 10)]
    norm_results['Z-Test_p_value'] = norm_results[z_columns].apply(lambda row: norm.sf(abs(row)).min(), axis=1)
    lognorm_results['Z-Test_p_value'] = lognorm_results[z_columns].apply(lambda row: norm.sf(abs(row)).min(), axis=1)

    # Set a minimum value to ensure display on a logarithmic scale
    min_value = 1e-15
    norm_results['Chi-Squared_p_value'] = norm_results['Chi-Squared_p_value'].apply(lambda x: max(x, min_value))
    lognorm_results['Chi-Squared_p_value'] = lognorm_results['Chi-Squared_p_value'].apply(lambda x: max(x, min_value))
    norm_results['Z-Test_p_value'] = norm_results['Z-Test_p_value'].apply(lambda x: max(x, min_value))
    lognorm_results['Z-Test_p_value'] = lognorm_results['Z-Test_p_value'].apply(lambda x: max(x, min_value))

    # Take the negative logarithm of the p-value
    norm_results['Chi-Squared_neg_log_p_value'] = -np.log10(norm_results['Chi-Squared_p_value'])
    lognorm_results['Chi-Squared_neg_log_p_value'] = -np.log10(lognorm_results['Chi-Squared_p_value'])
    norm_results['Z-Test_neg_log_p_value'] = -np.log10(norm_results['Z-Test_p_value'])
    lognorm_results['Z-Test_neg_log_p_value'] = -np.log10(lognorm_results['Z-Test_p_value'])

    # Visualized chi-square test results (vertical scale -log10(p-value))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(norm_results['ul'], norm_results['Chi-Squared_neg_log_p_value'], c='blue', label='Normal Distribution')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='Threshold -log10(0.05)')
    plt.title('Chi-Squared Test -log10(p-values) - Normal Distribution')
    plt.xlabel('ul')
    plt.ylabel('-log10(p-value)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(lognorm_results['ul'], lognorm_results['Chi-Squared_neg_log_p_value'], c='green', label='Lognormal Distribution')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='Threshold -log10(0.05)')
    plt.title('Chi-Squared Test -log10(p-values) - Lognormal Distribution')
    plt.xlabel('ul')
    plt.ylabel('-log10(p-value)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Visualized Z-test results (vertical scale -log10(p-value))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(norm_results['ul'], norm_results['Z-Test_neg_log_p_value'], c='blue', label='Normal Distribution')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='Threshold -log10(0.05)')
    plt.title('Z-Test -log10(p-values) - Normal Distribution')
    plt.xlabel('ul')
    plt.ylabel('-log10(p-value)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(lognorm_results['ul'], lognorm_results['Z-Test_neg_log_p_value'], c='green', label='Lognormal Distribution')
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='Threshold -log10(0.05)')
    plt.title('Z-Test -log10(p-values) - Lognormal Distribution')
    plt.xlabel('ul')
    plt.ylabel('-log10(p-value)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Visualization of SSD results (line graphs)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(norm_results['ul'], norm_results['SSD'], c='blue', label='Normal Distribution')
    plt.title('Sum of Squares Deviation (SSD) Results - Normal Distribution')
    plt.xlabel('ul')
    plt.ylabel('MOD')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(lognorm_results['ul'], lognorm_results['SSD'], c='green', label='Lognormal Distribution')
    plt.title('Sum of Squares Deviation (SSD) Results - Lognormal Distribution')
    plt.xlabel('ul')
    plt.ylabel('MOD')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Visualize REG results (line graphs, slopes and intercepts)
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(norm_results['ul'], norm_results['Reg_Slope'], c='blue', label='Normal Distribution')
    plt.axhline(y=1, color='red', linestyle='--', label='Expected Slope = 1')
    plt.title('Regression Measure (Slope) Results - Normal Distribution')
    plt.xlabel('ul')
    plt.ylabel('Slope')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(lognorm_results['ul'], lognorm_results['Reg_Slope'], c='green', label='Lognormal Distribution')
    plt.axhline(y=1, color='red', linestyle='--', label='Expected Slope = 1')
    plt.title('Regression Measure (Slope) Results - Lognormal Distribution')
    plt.xlabel('ul')
    plt.ylabel('Slope')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(norm_results['ul'], norm_results['Reg_Intercept'], c='blue', label='Normal Distribution')
    plt.axhline(y=0, color='red', linestyle='--',label='Expected Intercept = 0')
    plt.title('Regression Measure (Intercept) Results - Normal Distribution')
    plt.xlabel('ul')
    plt.ylabel('Intercept')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(lognorm_results['ul'], lognorm_results['Reg_Intercept'], c='green', label='Lognormal Distribution')
    plt.axhline(y=0, color='red', linestyle='--',label='Expected Intercept = 0')
    plt.title('Regression Measure (Intercept) Results - Lognormal Distribution')
    plt.xlabel('ul')
    plt.ylabel('Intercept')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    ul_values = 1000
    step = 1

    # Generate distribution data
    norm_df, lognorm_df = generate_distribution_data(ul_values, step)

    # Analyze distributed data
    norm_results, lognorm_results = analyze_distribution_data(norm_df, lognorm_df, ul_values)

    # Visualization data
    visualize_data(f'norm_results_{ul_values}.csv', f'lognorm_results_{ul_values}.csv')

    # Validation data
    validate_data(f'norm_results_{ul_values}.csv', f'lognorm_results_{ul_values}.csv')

if __name__ == "__main__":
    main()
