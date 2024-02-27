from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from matplotlib import use
use('tkAgg')

# Simulate the win margin  and win probability
vote_margin = np.linspace(-0.25, 0.25, 5000)
win_probability_t_plus_1 = np.random.binomial(1, norm.cdf(vote_margin, loc=0.1, scale=0.1))

# Calculate local averages within intervals of 0.005
intervals = np.arange(-0.25, 0.25, 0.005)
local_averages_x = []
local_averages_y = []

for start in intervals:
    end = start + 0.005
    mask = (vote_margin >= start) & (vote_margin < end)
    local_averages_x.append((start + end) / 2) # Midpoint of interval
    local_averages_y.append(win_probability_t_plus_1[mask].mean()) # Local average

# Fit a logistic regression model for the smooth curve
logistic_model = LogisticRegression()
logistic_model.fit(vote_margin.reshape(-1, 1), win_probability_t_plus_1)

# Generate predictions from the logistic model for the curve
logistic_curve = logistic_model.predict_proba(vote_margin.reshape(-1, 1))[:, 1]

# Plotting the figure
plt.figure(figsize=(10, 6))
plt.scatter(local_averages_x, local_averages_y, color='black', label='Local Average')
plt.plot(vote_margin, logistic_curve, color='black', linestyle='-', label='Logit fit')
plt.axvline(x=0, color='grey', linestyle='--')
plt.xlabel('Democratic Vote Share Margin of Victory, Election t')
plt.ylabel('Probability of Winning Election t+1')
plt.title('Candidate’s Probability of Winning Election t+1, by Margin of Victory in Election t')
plt.legend()
plt.ylim(0, 1)  # Probability range from 0 to 1
plt.show()
