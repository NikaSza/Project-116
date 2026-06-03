import numpy as np
import matplotlib.pyplot as plt

# 1. Define the mass range used in your analysis
m = np.linspace(5000, 5700, 1000)

# 2. Define the analytical function from the BDT Cut
# BDT achieved such high purity that it only has a Signal Gaussian component
def f_bdt(mass):
    amplitude = 6554.16
    mean = 5279.74
    sigma = 18.11
    
    signal = amplitude * np.exp(-0.5 * ((mass - mean) / sigma)**2)
    return signal

# 3. Define the analytical function from the NEW Manual Cuts
# Updated with the stable m - 5000 parameters
def f_manual(mass):
    amplitude = 5607.87
    mean = 5280.07
    sigma = 17.26
    c0 = 511.71
    c1 = -0.00391
    
    signal = amplitude * np.exp(-0.5 * ((mass - mean) / sigma)**2)
    background = c0 * np.exp(c1 * (mass - 5000.0))
    return signal + background

# 4. Separate out the manual background component for deeper comparison
# Updated with the stable m - 5000 parameters
def manual_background_only(mass):
    c0 = 511.71
    c1 = -0.00391
    return c0 * np.exp(c1 * (mass - 5000.0))

# =================================================================
# PLOTTING THE COMPARISON
# =================================================================
plt.figure(figsize=(10, 6))

# Plot the total BDT function
plt.plot(m, f_bdt(m), color='darkgreen', linestyle='-', linewidth=2.5, 
         label=r'BDT Selection (Pure Gauss: $\mu$=5279.74, $\sigma$=18.11)')

# Plot the total Manual Cut function
plt.plot(m, f_manual(m), color='darkred', linestyle='-', linewidth=2.5, 
         label=r'Manual Selection (Gauss+Exp: $\mu$=5280.07, $\sigma$=17.26)')

# Plot the leftover background from manual cuts to show why they differ
plt.plot(m, manual_background_only(m), color='red', linestyle='--', linewidth=1.5, 
         label='Leftover Background (Manual Cuts)')

# Formatting the plot
plt.title(r'Analytical Comparison: BDT vs. Shifted Manual Cuts for $B^0 \rightarrow K^{*0}\mu^+\mu^-$')
plt.xlabel('Invariant Mass $m$ [MeV]')
plt.ylabel('Analytical Amplitude (Events/Bin scale)')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =================================================================
# PRINT QUANTITATIVE COMPARISON REPORT
# =================================================================
print("="*60)
print("             METRIC COMPARISON SUMMARY")
print("="*60)
print(f"{'Metric':<25} | {'BDT Cut Method':<18} | {'Manual Cut Method':<18}")
print("-"*60)
print(f"{'Mass Mean (mu)':<25} | {'5279.74 MeV':<18} | {'5280.07 MeV':<18}")
print(f"{'Mass Resolution (sigma)':<25} | {'18.11 MeV':<18} | {'17.26 MeV':<18}")
print(f"{'Background Component':<25} | {'Completely Removed':<18} | {'Present (Exponential)':<18}")
print("="*60)