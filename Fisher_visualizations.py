import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from pathlib import Path

# %% 0) Global variables and helper functions
# ============================================

# --- Plotting Configuration ---
# Set to True to save the plots as PNG files
STORE_PLOTS = True
# Directory to save plots
PLOT_DIR = "fisher_plots_progressive"
# DPI for saved plots
PLOT_DPI = 300

if STORE_PLOTS:
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


# --- Simulation Parameters ---
# True, unknown parameter value
THETA_TRUE = 50
# Standard deviation for the measurement process
SIGMA_LOW_INFO = 10 # Wider distribution
SIGMA_HIGH_INFO = 2.5
# Number of sample measurements to show
N_SAMPLES = 8

def gaussian(x, mu, sigma):
    """Computes the Gaussian PDF."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# %% 1) Plot 1: The Core Problem - Measurement & Uncertainty (Progressive)
# =========================================================================

# --- Generate data once ---
mu = THETA_TRUE
sigma = SIGMA_LOW_INFO
x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
y_pdf = gaussian(x_range, mu, sigma)
np.random.seed(0) # for reproducibility
measurements = norm.rvs(loc=mu, scale=sigma, size=N_SAMPLES)

# --- Loop to create progressive plots ---
for stage in range(4):
    stage += 2
    with plt.xkcd():
        # --- Create the figure and axes for each stage ---
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Base Plotting Elements ---
        ax.set_title("The Measurement Problem", fontname='Humor Sans', fontsize=20, pad=20)
        ax.set_xlabel("Possible Measurement Values (x)", fontname='Humor Sans', fontsize=14)
        ax.set_ylabel(r"Probability Density $p(x|\theta)$", fontname='Humor Sans', fontsize=14)

        # --- Stage 1: Show the True (but unknown) Parameter ---
        if stage >= 1:
            ax.axvline(x=mu, color='r', linestyle='--', lw=2, label=r'True Value $\theta$')
            ax.annotate(r'True (but unknown)' + '\n' + r'Parameter $\theta$',
                        xy=(mu, 0.04), xytext=(mu - 3*sigma, 0.05),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="red", alpha=0.1))

        # --- Stage 2: Show the Probabilistic Measurement Process ---
        if stage >= 2:
            ax.plot(x_range, y_pdf, 'b', lw=2)
            ax.annotate(r'The measurement process is' + '\n' + r'probabilistic, following $p(x|\theta)$',
                        xy=(mu + .78*sigma, 0.03), xytext=(mu + 3*sigma, 0.04),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="skyblue", alpha=0.2))

        # --- Stage 3: Show a Few Actual Measurements ---
        if stage >= 3:
            ax.plot(measurements, [0]*N_SAMPLES, 'o', color='darkorange', markersize=8,
                    markeredgecolor='black', label='Measurements $x_i$')
            ax.annotate('Our Measurements, $x_i$',
                        xy=(measurements[1], 0), xytext=(measurements[1], 0.015),
                        arrowprops=dict(arrowstyle="->", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="orange", alpha=0.2))

        # --- Stage 4 & 5: State the Goal and the Fundamental Limit ---
        if stage >= 4:
            text_content = 'Our Goal: Estimate $\\theta$ from measurements $x_i$'
            if stage >= 5:
                text_content += '\n\nFundamental Limit:\nVar($\\hat{\\theta}$) $\\geq \\frac{1}{I(\\theta)}$'

            # MODIFIED: Changed position and alignment of the text box
            ax.text(0.03, 0.7, text_content,
                    transform=ax.transAxes, fontsize=14, fontname='Humor Sans',
                    verticalalignment='center', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

        # --- Common Styling for all stages ---
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_ylim(bottom=-0.002)
        ax.set_yticks([])
        for label in (ax.get_xticklabels()):
            label.set_fontname('Humor Sans')
            label.set_fontsize(11)

        plt.tight_layout()
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"01_measurement_problem_stage_{stage}.png"), dpi=PLOT_DPI)

        # Show the plot for the current stage
        plt.show()

        # Close the figure if it's not the final stage to avoid clutter
        if stage < 5:
            plt.close(fig)
# %% 2) Plot 2: The Likelihood Function (Progressive, Final Revisions)
# =========================================================================
X_OBS = 60
# This loop generates the plot in 3 distinct visual stages.
for stage in range(1, 4):
    with plt.xkcd():
        # --- Create the figure and axes for each stage ---
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Generate data for the plot ---
        x_obs = X_OBS
        sigma = SIGMA_LOW_INFO
        theta_range = np.linspace(x_obs - 4*sigma, x_obs + 4*sigma, 400)
        likelihood = gaussian(x_obs, theta_range, sigma)
        mle_theta = x_obs

        # --- MODIFIED: Set fixed axis limits for all stages ---
        # This prevents elements from jumping around between plots.
        y_max = gaussian(x_obs, x_obs, sigma)
        ax.set_xlim(theta_range[0], theta_range[-1])
        ax.set_ylim(-0.002, y_max * 1.3) # Added 30% padding on top

        # --- Base Plotting Elements ---
        ax.set_title("The Likelihood Function $\mathcal{L}(\\theta|x_{obs})$", fontname='Humor Sans', fontsize=20, pad=20)
        ax.set_xlabel("Possible Parameter Values ($\\theta$)", fontname='Humor Sans', fontsize=14)
        ax.set_ylabel(r"Likelihood", fontname='Humor Sans', fontsize=14)

        # --- Stage 1: Show the single measurement with the preferred text ---
        if stage >= 1:
            ax.plot([x_obs], [0], 'o', color='darkorange', markersize=9,
                    markeredgecolor='black', label=f'Observed Data $x_{{obs}} = {x_obs}$')
            # MODIFIED: Moved annotation to the top-left
            ax.annotate('We observe one data point...\n$x_{obs}$',
                        xy=(x_obs, 0.000), xytext=(x_obs - 2*sigma, 0.035),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="orange", alpha=0.2))

        # --- Stage 2: Add the likelihood function with the preferred text ---
        if stage >= 2:
            ax.plot(theta_range, likelihood, 'g', lw=2, label='Likelihood $\mathcal{L}(\\theta|x_{obs})$')
            ax.fill_between(theta_range, likelihood, color='green', alpha=0.1)
            # MODIFIED: Annotation moved lower and arrow target is now exactly on the curve
            target_x = x_obs + 1.5 * sigma
            target_y = gaussian(x_obs, target_x, sigma)
            ax.annotate('...and ask: "How likely is each\npossible value of $\\theta$?"',
                        xy=(target_x, target_y), xytext=(x_obs + 2.5*sigma, 0.03),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="green", alpha=0.1))

        # --- Stage 3: Identify the MLE with the preferred text in the bottom-left ---
        if stage >= 3:
            ax.axvline(x=mle_theta, color='darkgreen', linestyle=':', lw=2, label=r'MLE $\hat{\theta}$')
            ax.annotate('The most likely value $\\hat{\\theta}$\nis at the peak.',
                        xy=(mle_theta, gaussian(x_obs, mle_theta, sigma)),
                        xytext=(x_obs + 3.*sigma, 0.04),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.3))

        # --- Common Styling ---
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_yticks([])
        for label in (ax.get_xticklabels()):
            label.set_fontname('Humor Sans')
            label.set_fontsize(11)

        plt.tight_layout()
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"02_likelihood_final_final_stage_{stage}.png"), dpi=PLOT_DPI)

        # Show the plot for the current stage
        plt.show()

        # Close the figure if it's not the final stage
        if stage < 3:
            plt.close(fig)

# %% 3) Plot 3: Log-Likelihood Curvature IS Information (Progressive)
# ===================================================================

# This loop generates the plot in 5 distinct visual stages.
for stage in range(1, 4):
    with plt.xkcd():
        # --- Create the figure and axes for each stage ---
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Generate data ---
        x_obs = X_OBS
        theta_range = np.linspace(x_obs - 4*SIGMA_LOW_INFO, x_obs + 4*SIGMA_LOW_INFO, 400)

        # High Info (narrow distribution -> sharp log-likelihood)
        log_likelihood_high = np.log(gaussian(x_obs, theta_range, SIGMA_HIGH_INFO))

        # Low Info (wide distribution -> flat log-likelihood)
        log_likelihood_low = np.log(gaussian(x_obs, theta_range, SIGMA_LOW_INFO))

        # --- Set fixed axis limits for all stages ---
        ax.set_xlim(theta_range[0], theta_range[-1])
        ax.set_ylim(np.min(log_likelihood_low) * 1.1, np.max(log_likelihood_high) + 1)

        # --- Base Plotting Elements ---
        ax.set_title("The 'Sharpness' of Log-Likelihood is Information", fontname='Humor Sans', fontsize=20, pad=20)
        ax.set_xlabel("Possible Parameter Values ($\\theta$)", fontname='Humor Sans', fontsize=14)
        ax.set_ylabel(r"Log-Likelihood $\ln[\mathcal{L}(\theta|x_{obs})]$", fontname='Humor Sans', fontsize=14)
        ax.axvline(x=x_obs, color='k', linestyle=':', alpha=0.5) # MLE is at the peak

        # --- Stage 1: Plot the high-information curve ---
        if stage >= 1:
            ax.plot(theta_range, log_likelihood_high, 'r-', lw=2, label=f'High Info ($I \\propto 1/\\sigma^2$ is large)')
            ax.text(0.97, 0.4, # Adjusted position for clarity
                   'Definition:\n$I(\\theta) = -E\\left[ \\frac{\\partial^2}{\\partial \\theta^2} \\ln \\mathcal{L}(\\theta|x) \\right]$',
                   transform=ax.transAxes, fontsize=16, fontname='Humor Sans',
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

        # --- Stage 2: Add the low-information curve for comparison ---
        if stage >= 2:
            ax.plot(theta_range, log_likelihood_low, 'b--', lw=2, label=f'Low Info ($I \\propto 1/\\sigma^2$ is small)')
            ax.legend(prop={'family':'Humor Sans', 'size': 12})

        # --- Stage 3: Annotate the "Score" (the slope) ---
        if stage >= 3:
            ax.annotate('The slope (the "Score")\nchanges rapidly here.\nVery sensitive!',
                        xy=(x_obs - SIGMA_HIGH_INFO, np.log(gaussian(x_obs, x_obs - SIGMA_HIGH_INFO, SIGMA_HIGH_INFO))),
                        xytext=(x_obs - 3*SIGMA_HIGH_INFO, -2.5),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2, color='r'),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="red", alpha=0.1))
            ax.annotate('Slope here is flatter.\nLess sensitive.',
                        xy=(x_obs - SIGMA_LOW_INFO, np.log(gaussian(x_obs, x_obs - SIGMA_LOW_INFO, SIGMA_LOW_INFO))),
                        xytext=(x_obs - 3*SIGMA_LOW_INFO, -7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2, color='b'),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="skyblue", alpha=0.2))

        # --- Stage 4: Annotate the Curvature at the peak ---
        if stage >= 4:
            ax.annotate('Higher Curvature = More Information',
                        xy=(x_obs, 0), xytext=(x_obs + 2.5*SIGMA_LOW_INFO, -1.0),
                        arrowprops=dict(arrowstyle="-[,widthB=4.0,lengthB=1.0", lw=2, color='black'),
                        fontsize=14, fontname='Humor Sans', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.3))

        # --- Common Styling ---
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Humor Sans')
            label.set_fontsize(11)

        plt.tight_layout()
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"03_curvature_stage_{stage}.png"), dpi=PLOT_DPI)

        # Show the plot for the current stage
        plt.show()

        # Close the figure if it's not the final stage
        if stage < 5:
            plt.close(fig)
# %% 4) Plot 4: Comparing High and Low Information Measurements
# ==============================================================

for stage in range(1, 4):
    with plt.xkcd():
        # --- Create the figure and two subplots for comparison ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.patch.set_facecolor('white')

        # --- Axis 1: High Information Scenario ---
        ax1.set_facecolor('white')
        sigma_high = SIGMA_HIGH_INFO
        theta_range_high = np.linspace(X_OBS - 6*sigma_high, X_OBS + 6*sigma_high, 400)
        likelihood_high = gaussian(X_OBS, theta_range_high, sigma_high)

        ax1.plot(theta_range_high, likelihood_high, 'r-', lw=2)
        ax1.fill_between(theta_range_high, likelihood_high, color='red', alpha=0.1)
        ax1.set_title("High Information Measurement", fontname='Humor Sans', fontsize=16)
        ax1.set_xlabel("Possible Parameter Values ($\\theta$)", fontname='Humor Sans', fontsize=14)
        ax1.set_ylabel(r"Likelihood", fontname='Humor Sans', fontsize=14)
        ax1.axvline(x=X_OBS, color='k', linestyle=':', alpha=0.5)

        # --- Axis 2: Low Information Scenario ---
        ax2.set_facecolor('white')
        sigma_low = SIGMA_LOW_INFO
        theta_range_low = np.linspace(X_OBS - 4*sigma_low, X_OBS + 4*sigma_low, 400)
        likelihood_low = gaussian(X_OBS, theta_range_low, sigma_low)

        ax2.plot(theta_range_low, likelihood_low, 'b--', lw=2)
        ax2.fill_between(theta_range_low, likelihood_low, color='skyblue', alpha=0.2)
        ax2.set_title("Low Information Measurement", fontname='Humor Sans', fontsize=16)
        ax2.set_xlabel("Possible Parameter Values ($\\theta$)", fontname='Humor Sans', fontsize=14)
        ax2.axvline(x=X_OBS, color='k', linestyle=':', alpha=0.5)

        # --- Progressive Annotations ---
        if stage >= 1:
            ax1.text(0.5, 0.8,
                     'Precise Measurement\n(small $\\sigma$)',
                     transform=ax1.transAxes, fontsize=14, fontname='Humor Sans',
                     ha='center', bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
            ax2.text(0.5, 0.8,
                     'Imprecise Measurement\n(large $\\sigma$)',
                     transform=ax2.transAxes, fontsize=14, fontname='Humor Sans',
                     ha='center', bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.2))

        if stage >= 2:
            ax1.text(0.5, 0.6,
                     '$I(\\theta) = \\frac{1}{\\sigma^2}$ is LARGE',
                     transform=ax1.transAxes, fontsize=14, fontname='Humor Sans', ha='center')
            ax2.text(0.5, 0.6,
                     '$I(\\theta) = \\frac{1}{\\sigma^2}$ is SMALL',
                     transform=ax2.transAxes, fontsize=14, fontname='Humor Sans', ha='center')

        if stage >= 3:
            ax1.annotate("A measurement $x_{obs}$ from this\nprocess tells you a LOT\nabout where $\\theta$ could be.",
                         xy=(X_OBS + 0*sigma_high, 0.03), xytext=(X_OBS - 2*sigma_high, 0.05),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", lw=2),
                         fontsize=12, fontname='Humor Sans', ha='center',
                         bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.2))
            ax2.annotate("The same measurement from\nthis process is much less\ninformative. $\\theta$ could be\nanywhere in this wide range.",
                         xy=(X_OBS + 0*sigma_low, 0.03), xytext=(X_OBS + 2*sigma_low, 0.05),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", lw=2),
                         fontsize=12, fontname='Humor Sans', ha='center',
                         bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.2))

        # --- Overall Styling ---
        fig.suptitle("Not All Data is Created Equal", fontsize=22, fontname='Humor Sans')
        for ax in [ax1, ax2]:
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_ylim(bottom=0)
            ax.set_yticks([])
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname('Humor Sans')
                label.set_fontsize(11)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"04_comparison_stage_{stage}.png"), dpi=PLOT_DPI)

        # if stage == 3:
        #     plt.show()
        # else:
        #     plt.close(fig)

# %% 5) Plot 5: Accumulating Information (Progressive, Final Refinements)
# =======================================================================

# This loop generates the plot in 7 distinct visual stages.
for stage in range(1, 8):
    with plt.xkcd():
        # --- Create the figure and axes ---
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Generate data ---
        # MODIFIED: 6 measurements, more spread out
        x_obs = [35.0, 60.0, 42.0, 58.0, 51.0, 65.0]
        sigma = SIGMA_LOW_INFO
        theta_range = np.linspace(THETA_TRUE - 4*sigma, THETA_TRUE + 4*sigma, 500)
        colors = ['blue', 'green', 'purple', 'cyan', 'magenta', 'brown']

        # --- Set fixed axis limits for all stages ---
        ax.set_xlim(theta_range[0], theta_range[-1])
        ax.set_ylim(-0.05, 1.2)

        # --- Base Plotting Elements ---
        ax.set_title("Information Accumulates", fontname='Humor Sans', fontsize=20, pad=20)
        ax.set_xlabel("Possible Parameter Values ($\\theta$)", fontname='Humor Sans', fontsize=14)
        ax.set_ylabel(r"Normalized Likelihood", fontname='Humor Sans', fontsize=14)

        # --- Iteratively add measurements and update combined likelihood ---
        combined_likelihood = np.ones_like(theta_range)

        # Plot individual likelihoods first (dashed lines)
        num_measurements_to_show = min(stage, len(x_obs))
        for i in range(num_measurements_to_show):
            # MODIFIED: Add circle marker for each measurement
            ax.plot([x_obs[i]], [0], 'o', color='darkorange', markersize=8, markeredgecolor='black')

            likelihood_i = gaussian(x_obs[i], theta_range, sigma)
            # Update the running product for the combined likelihood
            combined_likelihood *= likelihood_i
            # Normalize for plotting aesthetics
            likelihood_i_norm = likelihood_i / np.max(likelihood_i)
            ax.plot(theta_range, likelihood_i_norm, '--', lw=2, color=colors[i],
                    label=f'$L( \\theta | x_{i+1}={x_obs[i]} )$')

        # Plot the combined likelihood (solid red line)
        if stage >= 1:
            combined_likelihood_norm = combined_likelihood / np.max(combined_likelihood)
            ax.plot(theta_range, combined_likelihood_norm, 'r-', lw=3, label='Combined Likelihood')
            ax.fill_between(theta_range, combined_likelihood_norm, color='red', alpha=0.15)


        # --- MODIFIED: Build up the I_total text formula progressively ---
        if stage >= 2:
            if stage == 2:
                info_text = '$I_{total} = I_1 + I_2$'
            elif stage == 3:
                info_text = '$I_{total} = I_1 + I_2 + I_3$'
            else: # stage >= 4
                info_text = f'$I_{{total}} = I_1 + I_2 + ... + I_{num_measurements_to_show}$'

            ax.text(0.03, 0.97,
                    'For independent measurements,\ninformation adds up:\n' + info_text,
                    transform=ax.transAxes, fontsize=14, fontname='Humor Sans',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

        # --- Stage 7: Add final annotation about sharpening ---
        if stage >= 7:
            peak_theta = theta_range[np.argmax(combined_likelihood_norm)]
            ax.annotate("Multiplying likelihoods\n(i.e., adding log-likelihoods)\nyields a much sharper peak!",
                        xy=(peak_theta, 1.0),
                        xytext=(peak_theta - 15, 0.7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.2))


        # --- Common Styling ---
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_yticks([0, 1])
        ax.legend(prop={'family':'Humor Sans', 'size': 11}, loc='upper right')

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Humor Sans')
            label.set_fontsize(11)

        plt.tight_layout()
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"05_accumulation_final_stage_{stage}.png"), dpi=PLOT_DPI)

        # Show the plot for the current stage
        plt.show()

        # Close the figure if it's not the final stage
        if stage < 7:
            plt.close(fig)

# %% 6) Plot 6: Visualizing a Single Estimation Trial (Revised)
# =============================================================

# --- Simulation Setup for this specific plot ---
N_LOW = 2    # Number of samples in the low-info case
N_HIGH = 10  # Number of samples in the high-info case

# Set a seed for this specific block to get a nice-looking example
np.random.seed(12)

# --- Generate samples for one trial ---
samples_low = norm.rvs(loc=THETA_TRUE, scale=SIGMA_LOW_INFO, size=N_LOW)-4.5+0.54
theta_hat_low = np.mean(samples_low)

samples_high = norm.rvs(loc=THETA_TRUE, scale=SIGMA_LOW_INFO, size=N_HIGH) + 3 - 0.08
theta_hat_high = np.mean(samples_high)

# --- Loop to generate plots progressively ---
# MODIFIED: Starts at stage 1, 4 stages total
for stage in range(1, 5):
    with plt.xkcd():
        # --- Create the figure and axes ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
        fig.patch.set_facecolor('white')
        fig.suptitle("The Process of Estimation", fontsize=22, fontname='Humor Sans')

        # --- Define data ranges and fixed limits ---
        theta_range = np.linspace(0, 100, 500) # Wider range for smooth curve
        # Likelihood for the mean has variance sigma^2 / N
        likelihood_low = gaussian(theta_hat_low, theta_range, SIGMA_LOW_INFO/np.sqrt(N_LOW))
        likelihood_high = gaussian(theta_hat_high, theta_range, SIGMA_LOW_INFO/np.sqrt(N_HIGH))
        y_max = np.max(likelihood_high)

        # Apply fixed limits to both axes
        for ax in [ax1, ax2]:
            # MODIFIED: Fixed x and y limits
            ax.set_xlim(30, 70)
            ax.set_ylim(0, y_max * 1.15)

        # --- Axis 1: Low-Sample Estimation Setup ---
        ax1.set_facecolor('white')
        ax1.set_title(f"One Trial with Few Samples (N={N_LOW})", fontname='Humor Sans', fontsize=16)
        ax1.set_xlabel("Value", fontname='Humor Sans', fontsize=14)
        ax1.set_ylabel(r"Probability/Likelihood", fontname='Humor Sans', fontsize=14)

        # --- Axis 2: High-Sample Estimation Setup ---
        ax2.set_facecolor('white')
        ax2.set_title(f"One Trial with More Samples (N={N_HIGH})", fontname='Humor Sans', fontsize=16)
        ax2.set_xlabel("Value", fontname='Humor Sans', fontsize=14)

        # --- Stage 1: Show the Ground Truth and the Drawn Samples ---
        if stage >= 1:
            for ax in [ax1, ax2]:
                ax.plot(theta_range, gaussian(theta_range, THETA_TRUE, SIGMA_LOW_INFO), 'k:', alpha=0.4, label='True PDF')
                ax.axvline(THETA_TRUE, color='r', linestyle='--', lw=2, label=f'True $\\theta$={THETA_TRUE:.1f}')
            # Plot the samples
            ax1.plot(samples_low, [y_max*0.02]*N_LOW, 'o', color='darkorange', markersize=8, markeredgecolor='black', label='Samples')
            ax2.plot(samples_high, [y_max*0.02]*N_HIGH, 'o', color='darkorange', markersize=8, markeredgecolor='black', label='Samples')

        # --- Stage 2: Form the Likelihood and the Estimate ---
        if stage >= 2:
            # Low-sample case
            ax1.plot(theta_range, likelihood_low, 'b-', lw=2, label='Likelihood')
            ax1.fill_between(theta_range, likelihood_low, color='skyblue', alpha=0.2)
            ax1.axvline(theta_hat_low, color='b', linestyle='-', lw=2, label=f'Estimate $\\hat{{\\theta}}$={theta_hat_low:.1f}')
            # High-sample case
            ax2.plot(theta_range, likelihood_high, 'r-', lw=2, label='Likelihood')
            ax2.fill_between(theta_range, likelihood_high, color='red', alpha=0.1)
            ax2.axvline(theta_hat_high, color='r', linestyle='-', lw=2, label=f'Estimate $\\hat{{\\theta}}$={theta_hat_high:.1f}')

        # --- Stage 3: Quantify the Error ---
        if stage >= 3:
           # --- Low-sample error annotation (CORRECTED) ---
            error_low = theta_hat_low - THETA_TRUE
            y_pos_low = np.max(likelihood_low) * 0.6
            ax1.annotate('', xy=(THETA_TRUE, y_pos_low), xytext=(theta_hat_low, y_pos_low),
                         arrowprops=dict(arrowstyle="<->", lw=2, color='purple'))
            ax1.text((THETA_TRUE + theta_hat_low) / 2, y_pos_low + 0.005,
                     f'Estimation Error\n({error_low:+.2f})',
                     ha='center', va='bottom', fontname='Humor Sans', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="purple", alpha=0.1))

            # --- High-sample error annotation (CORRECTED) ---
            error_high = theta_hat_high - THETA_TRUE
            y_pos_high = y_max * 0.6
            ax2.annotate('', xy=(THETA_TRUE, y_pos_high), xytext=(theta_hat_high, y_pos_high),
                         arrowprops=dict(arrowstyle="<->", lw=2, color='purple'))
            ax2.text((THETA_TRUE + theta_hat_high) / 2, y_pos_high + 0.005,
                     f'Estimation Error\n({error_high:+.2f})',
                     ha='center', va='bottom', fontname='Humor Sans', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="purple", alpha=0.1))


        # --- Stage 4: Add Legends ---
        if stage >= 4:
            ax1.legend(prop={'family':'Humor Sans', 'size': 11})
            ax2.legend(prop={'family':'Humor Sans', 'size': 11})

        # --- Overall Styling ---
        for ax in [ax1, ax2]:
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_yticks([])
            for label in (ax.get_xticklabels()):
                label.set_fontname('Humor Sans')
                label.set_fontsize(11)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"06_single_trial_revised_stage_{stage}.png"), dpi=PLOT_DPI)

        # Show the plot for the current stage
        plt.show()

        # Close the figure if it's not the final stage
        if stage < 4:
            plt.close(fig)

# %% 7) Plot 7: The Distribution of Many Estimates & The CRLB
# =============================================================

# --- Simulation Setup ---
N_EXPERIMENTS = 150 # Number of repeated experiments to run
N_LOW = 2           # Number of samples in the low-info case
N_HIGH = 10         # Number of samples in the high-info case

# --- Run Simulations ---
# This simulates repeating the process from the previous plot many times
theta_hats_low = []
theta_hats_high = []

np.random.seed(11)


for _ in range(N_EXPERIMENTS):
    # Low Info Case: Draw N_LOW samples and find the MLE (sample mean)
    samples_low = norm.rvs(loc=THETA_TRUE, scale=SIGMA_LOW_INFO, size=N_LOW)
    theta_hats_low.append(np.mean(samples_low))

    # High Info Case: Draw N_HIGH samples and find the MLE
    samples_high = norm.rvs(loc=THETA_TRUE, scale=SIGMA_LOW_INFO, size=N_HIGH)
    theta_hats_high.append(np.mean(samples_high))

# Convert to numpy arrays for easier stats
theta_hats_low = np.array(theta_hats_low)
theta_hats_high = np.array(theta_hats_high)


# --- Progressive Plotting ---
for stage in range(2, 5):
    with plt.xkcd():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5))
        fig.patch.set_facecolor('white')

        # Shared properties for both axes
        plot_range = [THETA_TRUE - 4*SIGMA_LOW_INFO, THETA_TRUE + 4*SIGMA_LOW_INFO]

        # --- Axis 1: Low Information (Few Samples) ---
        ax1.set_facecolor('white')
        ax1.hist(theta_hats_low, bins=30, density=True, color='skyblue', alpha=0.6)
        ax1.set_title(f"Estimates from Few Samples (N={N_LOW})", fontname='Humor Sans', fontsize=16)
        ax1.set_xlabel("Estimated Value ($\\hat{\\theta}$)", fontname='Humor Sans', fontsize=14)
        ax1.set_ylabel("Frequency of Estimate", fontname='Humor Sans', fontsize=14)
        ax1.set_xlim(plot_range)

        # --- Axis 2: High Information (More Samples) ---
        ax2.set_facecolor('white')
        ax2.hist(theta_hats_high, bins=30, density=True, color='red', alpha=0.4)
        ax2.set_title(f"Estimates from More Samples (N={N_HIGH})", fontname='Humor Sans', fontsize=16)
        ax2.set_xlabel("Estimated Value ($\\hat{\\theta}$)", fontname='Humor Sans', fontsize=14)
        ax2.set_xlim(plot_range)

        # Apply fixed limits to both axes
        for ax in [ax1, ax2]:
            ax.set_xlim(20, 80)
            # ax.set_ylim(0, y_max * 1.15


        # --- Progressive Reveals ---
        # Stage 1: Add true value line
        if stage >= 1:
            for ax in [ax1, ax2]:
                ax.axvline(THETA_TRUE, color='k', linestyle='--', lw=2, label=f'True $\\theta$ = {THETA_TRUE:.1f}')

        # Stage 2: Show the variance of the estimates
        if stage >= 2:
            var_low = np.var(theta_hats_low)
            var_high = np.var(theta_hats_high)
            ax1.text(0.97, 0.97, f'Observed Var($\\hat{{\\theta}}$) = {var_low:.2f}', transform=ax1.transAxes, ha='right', va='top', fontsize=12, fontname='Humor Sans', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
            ax2.text(0.97, 0.97, f'Observed Var($\\hat{{\\theta}}$) = {var_high:.2f}', transform=ax2.transAxes, ha='right', va='top', fontsize=12, fontname='Humor Sans', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

        # Stage 3: Annotate the clear difference in precision
        if stage >= 3:
             ax1.annotate("Our estimates are\nare more spread out!",
                         xy=(THETA_TRUE - 1. * SIGMA_LOW_INFO, 0.02),
                         xytext=(THETA_TRUE - 1.6 * SIGMA_LOW_INFO, 0.04),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2, color='b'),
                         fontsize=12, fontname='Humor Sans', ha='center',
                         bbox=dict(boxstyle="round,pad=0.4", fc="skyblue", alpha=0.2))
             ax2.annotate("More precise!\nEstimates are tightly\nclustered around true value.",
                         xy=(THETA_TRUE + 0.3 * SIGMA_LOW_INFO, 0.02),
                         xytext=(THETA_TRUE + 1.4 * SIGMA_LOW_INFO, 0.13),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=2, color='r'),
                         fontsize=12, fontname='Humor Sans', ha='center',
                         bbox=dict(boxstyle="round,pad=0.4", fc="red", alpha=0.1))

        fig.text(0.5, -0.05,  # place holder in white to keep plot in place
                 "The Cramer-Rao Lower Bound (CRLB) formalizes this: Var($\\hat{\\theta}$) $\\geq \\frac{1}{I_{total}} = \\frac{\\sigma^2}{N}$.\nOur estimator is efficient, so its variance meets this bound!",
                 ha='center', color='white', va='bottom', fontsize=18, fontname='Humor Sans', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.0))

        # Stage 4: Reveal the CRLB
        if stage >= 4:
            crlb_low = SIGMA_LOW_INFO**2 / N_LOW
            crlb_high = SIGMA_LOW_INFO**2 / N_HIGH
            ax1.axhline(0, xmin=0.3, xmax=0.7, color='purple', lw=4, solid_capstyle='round', label=f'CRLB = {crlb_low:.2f}')
            ax2.axhline(0, xmin=0.4, xmax=0.6, color='purple', lw=4, solid_capstyle='round', label=f'CRLB = {crlb_high:.2f}')
            ax1.legend(prop={'family':'Humor Sans', 'size': 12})
            ax2.legend(prop={'family':'Humor Sans', 'size': 12})


            fig.text(0.5, 0.03,
                     "The Cramer-Rao Lower Bound (CRLB) formalizes this: Var($\\hat{\\theta}$) $\\geq \\frac{1}{I_{total}} = \\frac{\\sigma^2}{N}$.\nOur estimator is efficient, so its variance meets this bound!",
                     ha='center', va='bottom', fontsize=18, fontname='Humor Sans', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.15))

        # --- Overall Styling ---
        fig.suptitle("Repeating the Experiment Reveals the Error", fontsize=22, fontname='Humor Sans')
        for ax in [ax1, ax2]:
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.set_yticks([])
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname('Humor Sans')
                label.set_fontsize(11)

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        # plt.tight_layout()
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"07_crlb_stage_{stage}.png"), dpi=PLOT_DPI)

        plt.show(fig)

# %% 8) Plot 8: Optimizing Experiments via the CRLB (Final Version)
# ================================================================

# This plot frames experimental design as an optimization problem:
# To minimize the CRLB on our estimate, we must maximize information.
# For this system, that means maximizing the total photons collected.

for stage in range(1, 5):
    with plt.xkcd():
        # --- Create the figure and axes ---
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Generate data ---
        # The x-axis is the total number of photons collected, N = λ*T
        total_photons = np.linspace(1, 100, 400)
        # The y-axis is the lower bound on relative error, derived from the CRLB
        lower_bound_rel_error = 1 / np.sqrt(total_photons)

        # --- Set fixed axis limits ---
        ax.set_xlim(0, total_photons[-1])
        ax.set_ylim(0, lower_bound_rel_error[0] * 1.1)

        # --- Stage 1: Show the fundamental performance limit ---
        if stage >= 1:
            ax.plot(total_photons, lower_bound_rel_error, 'm-', lw=3)
            ax.set_title("Optimizing an Experiment by Minimizing the CRLB", fontname='Humor Sans', fontsize=20, pad=20)
            ax.set_xlabel("Total Photons Collected ($N = \\lambda \\times T$)", fontname='Humor Sans', fontsize=14)
            # MODIFIED: Formalized the y-axis label
            ax.set_ylabel("Lower Bound on Relative Error", fontname='Humor Sans', fontsize=14)

        # --- Stage 2: Highlight a suboptimal experiment ---
        if stage >= 2:
            photons_A = 10
            error_A = 1 / np.sqrt(photons_A)
            ax.plot(photons_A, error_A, 'o', color='skyblue', markersize=10, markeredgecolor='k')
            ax.annotate(f"Experiment A: {photons_A} photons\nRelative error is high (~{error_A:.2f})",
                        xy=(photons_A, error_A), xytext=(photons_A + 20, error_A + 0.3),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="skyblue", alpha=0.2))

        # --- Stage 3: Highlight an optimized experiment ---
        if stage >= 3:
            photons_B = 80
            error_B = 1 / np.sqrt(photons_B)
            ax.plot(photons_B, error_B, 'o', color='red', markersize=10, markeredgecolor='k')
            ax.annotate(f"Experiment B: {photons_B} photons\nRelative error is low (~{error_B:.2f})",
                        xy=(photons_B, error_B), xytext=(photons_B - 20, error_B + 0.3),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=2),
                        fontsize=12, fontname='Humor Sans', ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.1))

        # --- Stage 4: Add the final conclusion on optimization ---
        if stage >= 4:
            # MODIFIED: Formalized the conclusion text
            ax.text(0.97, 0.97,
                    'Goal: Minimize the CRLB\n(i.e., Maximize Fisher Information)\n\nFor this system, this requires maximizing\nthe total photons collected ($N_{photons}$)',
                    transform=ax.transAxes, fontsize=14, fontname='Humor Sans',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))


        # --- Styling ---
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # Invert y-axis to frame as minimization problem (lower is better)
        # ax.invert_yaxis() # Optional: could be interesting but maybe confusing
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Humor Sans')
            label.set_fontsize(11)

        plt.tight_layout()
        if STORE_PLOTS:
            plt.savefig(os.path.join(PLOT_DIR, f"08_optimization_stage_{stage}.png"), dpi=PLOT_DPI)

        if stage == 4:
            plt.show()
        else:
            plt.close(fig)