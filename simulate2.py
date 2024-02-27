import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use
use('tkAgg')

# Simulate the data
np.random.seed(0)  # For reproducibility
vote_margin = np.linspace(-0.25, 0.25, 500)
past_victories = vote_margin**3 + np.random.normal(0, 0.1, vote_margin.shape[0])

# Calculate local averages within intervals of 0.005
intervals = np.arange(-0.25, 0.25, 0.005)
local_averages_x = []
local_averages_y = []

for start in intervals:
    end = start + 0.005
    mask = (vote_margin >= start) & (vote_margin < end)
    local_averages_x.append((start + end) / 2)  # Midpoint of interval
    if np.any(mask):
        local_averages_y.append(past_victories[mask].mean())  # Local average
    else:
        local_averages_y.append(np.nan)

# Remove NaNs from local averages for fitting
clean_indices = ~np.isnan(local_averages_y)
local_averages_x_clean = np.array(local_averages_x)[clean_indices]
local_averages_y_clean = np.array(local_averages_y)[clean_indices]

# Fit a polynomial model for the smooth curve
coefficients = np.polyfit(local_averages_x_clean, local_averages_y_clean, 3)
polynomial = np.poly1d(coefficients)

# Generate predictions from the polynomial model for the curve
poly_curve = polynomial(vote_margin)

# Plotting the figure
plt.figure(figsize=(10, 6))
plt.scatter(local_averages_x, local_averages_y, color='black', label='Local Average')
plt.plot(vote_margin, poly_curve, color='black', linestyle='-', label='Polynomial fit')
plt.axvline(x=0, color='grey', linestyle='--')
plt.xlabel('Democratic Vote Share Margin of Victory, Election t')
plt.ylabel('No. of Past Victories of Election')
plt.title('Democratic Vote Share Margin of Victory, Election t')
plt.legend()
plt.ylim(0, max(past_victories) + 0.5)  # Adjust the y-axis range
plt.show()
