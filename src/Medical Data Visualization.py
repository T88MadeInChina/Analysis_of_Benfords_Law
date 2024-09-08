import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Read the sub_medical_data_result.csv file or entire_medical_data_result.csv
df = pd.read_csv('sub_medical_data_results.csv')

# Convert CODE columns to string types
df['CODE'] = df['CODE'].astype(str)

# Calculate the maximum of the 9 Z-test values
z_columns = [f'Z-Test_{i}' for i in range(1, 10)]
df['Z-Test_Max'] = df[z_columns].max(axis=1)

# Set a minimum p-value
df['Chi-Squared'] = df['Chi-Squared'].apply(lambda x: max(x, 1e-20))
df['Z-Test_Max'] = df['Z-Test_Max'].apply(lambda x: max(x, 1e-20))

# Visualized chi-square test results
plt.figure(figsize=(12, 6))
plt.bar(df['CODE'], df['Chi-Squared'], color='blue')
plt.axhline(y=15.51, color='red', linestyle='-', label='Threshold log10(15.51)')
# plt.title('Chi-Square Test Results Of Entire Dataset')
plt.title('Chi-Square Test Results Of Subdata Set')
plt.xlabel('Medical Data Code')
plt.ylabel('Chi-Squared (Log Scale)')
plt.yscale('log')
plt.xticks(rotation=90)
plt.legend()

# 可视化Z检验结果
plt.figure(figsize=(12, 6))
plt.bar(df['CODE'], df['Z-Test_Max'], color='green')
plt.axhline(y=1.96, color='red', linestyle='-', label='Threshold log10(1.96)')
# plt.title('Z-Test Results (Max of 9 Z-Scores) Of Entire Dataset')
plt.title('Z-Test Results (Max of 9 Z-Scores) Of Subdata Set')

plt.xlabel('Medical Data Code')
plt.ylabel('Z-Test Max (Log Scale)')
plt.yscale('log')
plt.xticks(rotation=90)
plt.legend()

plt.tight_layout()
plt.show()

# Visualize SSD results
plt.figure(figsize=(12, 6))
plt.plot(df['CODE'], df['SSD'], marker='o', color='blue', label='SSD')
# plt.title('Sum of Squares Deviation (SSD) Results Of Entire Dataset')
plt.title('Sum of Squares Deviation (SSD) Results Of Subdata Set')

plt.xlabel('Medical Data Code')
plt.ylabel('SSD')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# Visualize REG results (slope and intercept)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['CODE'], df['Reg_Slope'], marker='o', color='blue', label='Slope')
plt.axhline(y=1, color='red', linestyle='-', label='Expected Slope = 1')
plt.title('Regression Measure (Slope) Results Of Entire Dataset')
# plt.title('Regression Measure (Slope) Results Of Subdata Set')
plt.xlabel('Medical Data Code')
plt.ylabel('Slope')
plt.xticks(rotation=90)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['CODE'], df['Reg_Intercept'], marker='o', color='green', label='Intercept')
plt.axhline(y=0, color='red', linestyle='-', label='Expected Intercept = 0')
plt.title('Regression Measure (Intercept) Results')
plt.xlabel('Medical Data Code')
plt.ylabel('Intercept')
plt.xticks(rotation=90)
plt.legend()

plt.tight_layout()
plt.show()
