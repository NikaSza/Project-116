# 1. Define the mathematical models (Updated with m - 5000 shift)
def gaussian_signal(x, amplitude, mean, stddev):
    """The signal peak"""
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

def exponential_background(x, c0, c1):
    """The combinatorial background shifted to avoid underflow/overflow"""
    return c0 * np.exp(c1 * (x - 5000.0))

def combined_model(x, amplitude, mean, stddev, c0, c1):
    """Signal + Background"""
    return gaussian_signal(x, amplitude, mean, stddev) + exponential_background(x, c0, c1)

def fit_and_plot_mass(mass_data, bins=300, mass_min=4000, mass_max=6000):
    # 2. Create the histogram data to fit
    counts, bin_edges = np.histogram(mass_data, bins=bins, range=(mass_min, mass_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    nonzero_mask = counts > 0
    x_data = bin_centers[nonzero_mask]
    y_data = counts[nonzero_mask]

    # 3. Provide initial guesses for the fit parameters
    # [amplitude, mean, stddev, background_c0, background_c1]
    # FIX: counts[0] corresponds to roughly mass=5000, which aligns perfectly with our shift!
    initial_guess = [max(counts), 5279, 20, counts[0], -0.001]
    
    # 4. Perform the curve fit
    try:
        popt, pcov = curve_fit(combined_model, x_data, y_data, p0=initial_guess)
        amplitude, mean, stddev, c0, c1 = popt
        
        # 5. Calculate the Number of Signal Events
        n_signal_events = (amplitude * np.abs(stddev) * np.sqrt(2 * np.pi)) / bin_width
        
        # Calculate errors on the parameters
        perr = np.sqrt(np.diag(pcov))
        n_signal_error = (perr[0] * np.abs(stddev) * np.sqrt(2 * np.pi)) / bin_width 
        
        print(f"--- Fit Results ---")
        print(f"B0 Mass Mean: {mean:.2f} MeV/c^2")
        print(f"Mass Resolution (StdDev): {np.abs(stddev):.2f} MeV/c^2")
        print(f"Number of Signal Events: {n_signal_events:.0f} ± {n_signal_error:.0f}")
        
        # Print the Analytical Extracted Function for Manual Cuts
        print("\n" + "="*50)
        print("EXTRACTED ANALYTICAL FUNCTION (MANUAL CUTS)")
        print("="*50)
        # FIX: Explicitly printing the shifted function format
        print(f"Post-manual-cut Function f(m):\n  {amplitude:.2f} * exp(-0.5*((m - {mean:.2f})/{np.abs(stddev):.2f})^2) + {c0:.2f} * exp({c1:.5f} * (m - 5000))")
        print("="*50 + "\n")
        
    except RuntimeError:
        print("Error: Curve fit failed to converge. Try adjusting the initial guesses or cuts.")
        return

    # 6. Plotting the results
    x_plot = np.linspace(mass_min, mass_max, 500)
    
    plt.figure(figsize=(8, 6))
    
    # Plot original data as points with error bars (Poisson errors)
    plt.errorbar(x_data, y_data, yerr=np.sqrt(y_data), fmt='o', color='black', label='Data', markersize=4)
    
    # Plot the total fit
    plt.plot(x_plot, combined_model(x_plot, *popt), color='red', linewidth=2, label='Total Fit')
    
    # Plot the signal component
    plt.plot(x_plot, gaussian_signal(x_plot, amplitude, mean, stddev), color='blue', linestyle='--', label='Signal (Gaussian)')
    
    # Plot the background component
    plt.plot(x_plot, exponential_background(x_plot, c0, c1), color='green', linestyle='--', label='Background (Exponential)')
    
    plt.title("B0 Mass Distribution Fit")
    plt.xlabel("B0_MM [MeV/c^2]")
    plt.ylabel(f"Events / {bin_width:.1f} MeV/c^2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()