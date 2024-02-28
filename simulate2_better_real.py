import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.polynomial.polynomial import Polynomial
from matplotlib import use

use('tkAgg')

# Function to simulate past victories with some noise
def simulate_victories_with_noise(margin, loc=0.1, scale=0.1, elections=10, offset=0):
    prob_win = norm.cdf(margin, loc=loc, scale=scale)
    victories = np.random.binomial(elections, prob_win)
    # Introduce random noise
    noise = np.random.uniform(-0.3, 0.3, size=victories.shape)
    return victories + noise + offset

# Simulate the data
vote_margins = np.linspace(-0.25, 0.25, 5000)
accumulated_victories_with_noise = simulate_victories_with_noise(vote_margins, elections=5)

# Calculate local averages within intervals of 0.005
intervals = np.arange(-0.25, 0.25, 0.005)
local_averages_x = []
local_averages_y = []

for start in intervals:
    end = start + 0.005
    mask = (vote_margins >= start) & (vote_margins < end)
    local_averages_x.append((start + end) / 2)  # Midpoint of interval
    if start < 0:
        local_averages_y.append(accumulated_victories_with_noise[mask].mean())
    else:
        local_averages_y.append(accumulated_victories_with_noise[mask].mean() + 0.3)

# Fit polynomials to the left and right side
left_side_mask = np.array(local_averages_x) < 0
right_side_mask = np.array(local_averages_x) >= 0

left_polynomial_model = Polynomial.fit(np.array(local_averages_x)[left_side_mask],
                                       np.array(local_averages_y)[left_side_mask], deg=3)
right_polynomial_model = Polynomial.fit(np.array(local_averages_x)[right_side_mask],
                                        np.array(local_averages_y)[right_side_mask], deg=3)

# Generate predictions from the polynomial models for the curves
left_polynomial_curve = left_polynomial_model(vote_margins[vote_margins < 0])
right_polynomial_curve = right_polynomial_model(vote_margins[vote_margins >= 0])

# Plotting the results
plt.figure(figsize=(10, 5))
plt.scatter(local_averages_x, local_averages_y, color='black', s=10, alpha=0.5, label='Local Averages')
plt.plot(vote_margins[vote_margins < 0], left_polynomial_curve, color='black', linewidth=2)
plt.plot(vote_margins[vote_margins >= 0], right_polynomial_curve, color='black', linewidth=2, label='Polynomial Fit')
plt.axvline(x=0, color='grey', linestyle='--')
plt.xlabel('Democratic Vote Share Margin of Victory, Election t')
plt.ylabel('No. of Past Victories at Election t')
plt.title("Adapted Figure 2b: Candidate's Accumulated Number of Past Election Victories (Local Averages)")
plt.legend()
plt.grid(True)
plt.xlabel('Democratic Vote Share Margin of Victory, Election t')
plt.ylabel('No. of Past Victories at Election t')
plt.title("Figure 2b: Candidate's Accumulated Number of Past Election Victories")
plt.xlim([-0.25, 0.25])
plt.ylim([0, 5])
plt.xticks(np.arange(-0.25, 0.26, 0.05))
plt.yticks(np.arange(0, 5.1, 0.5))
plt.gca().set_facecolor('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()