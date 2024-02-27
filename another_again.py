import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from matplotlib import use
import matplotlib.pyplot as plt
from scipy import stats
use('tkAgg')

df = pd.read_csv('./output_file.csv')
import numpy as np

# Assuming df is your DataFrame after loading the CSV
# Ensure the 'Candidate Identifier' is correctly created (adapt this part based on your dataset)
df["CANDIDATE'S NAME"] = df["CANDIDATE'S NAME"].fillna('').astype(str)
df['Candidate Identifier'] = df['ICPSR STATE CODE'].astype(str) + df["CANDIDATE'S NAME"].str.replace('.', '').str.upper().apply(lambda x: x[:5] if len(x) >= 5 else x) + df["CANDIDATE'S NAME"].str[0]

df = df[df["CANDIDATE PERCENT FROM VICTOR"] != 9999] # remove the rows that are missing this data


# Sort by 'Candidate Identifier' and 'YEAR OF ELECTION' for sequential calculation
df.sort_values(by=['Candidate Identifier', 'YEAR OF ELECTION'], inplace=True)

# Calculate 'Vote Share t+1' for each candidate
df['Vote Share t+1'] = df.groupby('Candidate Identifier')["CANDIDATE'S PERCENT"].shift(-1)
df.dropna(subset=['Vote Share t+1'], inplace=True)  # Optional: Drop rows without a 'Vote Share t+1'

df['Normalized_Margin_of_Victory'] = df['MARGIN OF VICTORY'] / df['TOTAL VOTE CAST IN ELEC']
df['Victory_Election_t'] = (df['ELECTION OUTCOME'] == 1).astype(int)


# Generating polynomial and interaction terms based on the normalized margin of victory and victory dummy
for i in range(2, 5):  # Generating squared to quartic terms
    df[f'Normalized_Margin_of_Victory_{i}'] = df['Normalized_Margin_of_Victory'] ** i
    df[f'Victory_Margin_{i}'] = df['Victory_Election_t'] * df[f'Normalized_Margin_of_Victory_{i}']

# Ensure all column names are correctly formatted for statsmodels formula (replace spaces with underscores if necessary)
# Ensure 'Vote Share t+1' is calculated and exists in the DataFrame
df['Vote_Share_t1'] = df.groupby('Candidate Identifier')["CANDIDATE'S PERCENT"].shift(-1)
df.dropna(subset=['Vote_Share_t1'], inplace=True)  # Drop rows without a 'Vote Share t+1'

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
df = df[np.abs(stats.zscore(df['Vote_Share_t1'])) < 3]
df = df[np.abs(stats.zscore(df['Normalized_Margin_of_Victory'])) < 3]
print(df['Vote_Share_t1'].describe())
print(df['Normalized_Margin_of_Victory'].describe())



# Assuming df is loaded and preprocessed

# Bin the data by vote share margin
bin_width = 0.005
df['Margin_Bin'] = np.floor(df['Normalized_Margin_of_Victory'] / bin_width) * bin_width

# Calculate the mean outcome for each bin for visualization
bin_means = df.groupby('Margin_Bin')['Victory_Election_t'].mean().reset_index()

# Visualize the binned averages
plt.figure(figsize=(10, 6))
plt.scatter(bin_means['Margin_Bin'], bin_means['Victory_Election_t'], alpha=0.5, label='Binned average')

# Perform logistic regression using the original vote share margin (not binned)
# Prepare the feature matrix and target vector
X = df[['Normalized_Margin_of_Victory']]
y = df['Victory_Election_t']

# Fit the logistic regression model
logit_model = LogisticRegression()
logit_model.fit(X, y)

# Generate predictions for the logistic curve
X_pred = np.linspace(df['Normalized_Margin_of_Victory'].min(), df['Normalized_Margin_of_Victory'].max(), 300).reshape(-1, 1)
y_pred = logit_model.predict_proba(X_pred)[:, 1]

# Overlay the logistic regression curve
plt.plot(X_pred, y_pred, color='red', label='Logit Fit')

# Highlight the discontinuity at 0
plt.axvline(x=0, color='green', linestyle='--', label='Zero Margin (Discontinuity)')

plt.title('Estimated Probability of Winning Election t+1 vs. Vote Share Margin of Victory in Election t')
plt.xlabel('Normalized Margin of Victory in Election t')
plt.ylabel('Probability of Winning Election t+1')
plt.legend()
plt.grid(True)
plt.show()


'''
# Normalize and create interaction terms
df['Normalized_Margin_of_Victory'] = df['MARGIN OF VICTORY'] / df['TOTAL VOTE CAST IN ELEC']
df['Victory_Election_t'] = (df['ELECTION OUTCOME'] == 1).astype(int)

for i in range(2, 5):  # Squared to quartic terms
    df[f'Normalized_Margin_of_Victory_{i}'] = df['Normalized_Margin_of_Victory'] ** i
    df[f'Victory_Margin_{i}'] = df['Victory_Election_t'] * df[f'Normalized_Margin_of_Victory_{i}']

# Correctly format column names for statsmodels (if not already done)
df.columns = [col.replace(' ', '_') for col in df.columns]

# Model formula (ensure it matches the DataFrame's column names)
formula = ('Vote_Share_t1 ~ Normalized_Margin_of_Victory + '
           'Normalized_Margin_of_Victory_2 + Normalized_Margin_of_Victory_3 + Normalized_Margin_of_Victory_4 + '
           'Victory_Election_t + Victory_Margin_2 + Victory_Margin_3 + Victory_Margin_4')

# Fit the model
model = smf.ols(formula=formula, data=df).fit()

# Print the model summary
print(model.summary())

df['Winning_t1'] = (df['Vote_Share_t1'] > 50).astype(int)

df['Margin_Bin'] = pd.cut(df['Normalized_Margin_of_Victory'], bins=50, labels=False)

# Calculate local averages within each bin
local_avg = df.groupby('Margin_Bin').agg({'Winning_t1': 'mean', 'Normalized_Margin_of_Victory': 'mean'})

# Prepare for plotting
x_local_avg = local_avg['Normalized_Margin_of_Victory']
y_local_avg = local_avg['Winning_t1']

# Fit a logistic regression model for parametric fit
X = df[['Normalized_Margin_of_Victory']]  # Predictor
y = df['Winning_t1']  # Binary outcome

logit_model = LogisticRegression()
logit_model.fit(X, y)

# Generate predictions for a smooth curve
X_predict = np.linspace(df['Normalized_Margin_of_Victory'].min(), df['Normalized_Margin_of_Victory'].max(), 300).reshape(-1, 1)
y_predict_proba = logit_model.predict_proba(X_predict)[:, 1]

plt.figure(figsize=(10, 6))

# Plot local averages
plt.scatter(x_local_avg, y_local_avg, color='blue', label='Local Averages')

# Plot logistic regression fit
plt.plot(X_predict.flatten(), y_predict_proba, color='red', label='Parametric Fit (Logistic Regression)')

plt.axvline(x=0, color='black', linestyle='--', label='Victory Threshold')
plt.xlabel('Normalized Margin of Victory in Election t')
plt.ylabel('Probability of Winning Election t+1')
plt.title("Candidate's Probability of Winning t+1 by Margin of Victory in t")
plt.legend()
plt.show()
'''
