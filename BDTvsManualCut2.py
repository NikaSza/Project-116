import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Define the mass range used in your analysis
mass_min, mass_max = 5000.0, 5700.0
m = np.linspace(mass_min, mass_max, 1000)

# =================================================================
# ANALYTICAL FUNCTIONS (DEFINED AS STANDARD MATH FUNCTIONS FOR INTEGRATION)
# =================================================================

# New BDT Components (Gauss + Exp)
def bdt_signal_only(mass):
    amplitude, mean, sigma = 6483.79, 5279.83, 17.59
    return amplitude * np.exp(-0.5 * ((mass - mean) / sigma)**2)

def bdt_background_only(mass):
    b_bkg, c_bkg = 402.06, 0.00442
    return b_bkg * np.exp(-c_bkg * (mass - 5000.0))

def f_bdt(mass):
    return bdt_signal_only(mass) + bdt_background_only(mass)


# Manual Cut Components
def manual_signal_only(mass):
    amplitude, mean, sigma = 5607.87, 5280.07, 17.26
    return amplitude * np.exp(-0.5 * ((mass - mean) / sigma)**2)

def manual_background_only(mass):
    c0, c1 = 511.71, -0.00391
    return c0 * np.exp(c1 * (mass - 5000.0))

def f_manual(mass):
    return manual_signal_only(mass) + manual_background_only(mass)


# =================================================================
# CALCULATING AREAS UNDER THE CURVES (INTEGRALS)
# =================================================================

# BDT Areas
bdt_sig_area, _ = quad(bdt_signal_only, mass_min, mass_max)
bdt_bkg_area, _ = quad(bdt_background_only, mass_min, mass_max)
bdt_total_area = bdt_sig_area + bdt_bkg_area
bdt_purity = (bdt_sig_area / bdt_total_area) * 100

# Manual Cut Areas
manual_sig_area, _ = quad(manual_signal_only, mass_min, mass_max)
manual_bkg_area, _ = quad(manual_background_only, mass_min, mass_max)
manual_total_area = manual_sig_area + manual_bkg_area
manual_purity = (manual_sig_area / manual_total_area) * 100


# =================================================================
# PLOTTING THE COMPARISON (WITH FILLED BACKGROUND AREAS TO VISUALIZE)
# =================================================================
plt.figure(figsize=(11, 6.5))

# Plot total composite curves
plt.plot(m, f_bdt(m), color='darkgreen', linestyle='-', linewidth=2.5, 
         label=r'BDT Selection (Gauss+Exp: $\mu$=5279.83, $\sigma$=17.59)')

plt.plot(m, f_manual(m), color='darkred', linestyle='-', linewidth=2.5, 
         label=r'Manual Selection (Gauss+Exp: $\mu$=5280.07, $\sigma$=17.26)')

# Plot and shade the BDT background area
plt.plot(m, bdt_background_only(m), color='green', linestyle=':', linewidth=1.5, 
         label='Residual Background (BDT)')
plt.fill_between(m, 0, bdt_background_only(m), color='green', alpha=0.08, label='Background Area (BDT)')

# Plot and shade the manual background area
plt.plot(m, manual_background_only(m), color='red', linestyle='--', linewidth=1.5, 
         label='Leftover Background (Manual Cuts)')
plt.fill_between(m, 0, manual_background_only(m), color='red', alpha=0.08, label='Background Area (Manual)')

# Formatting the plot
plt.title(r'Analytical & Area Comparison: Updated BDT vs. Manual Cuts ($B^0 \rightarrow K^{*0}\mu^+\mu^-$)')
plt.xlabel('Invariant Mass $m$ [MeV]')
plt.ylabel('Analytical Amplitude (Events/Bin scale)')
plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =================================================================
# PRINT QUANTITATIVE COMPARISON REPORT
# =================================================================
print("="*75)
print(f"{'METRIC':<28} | {'BDT CUT METHOD':<20} | {'MANUAL CUT METHOD':<20}")
print("="*75)
print(f"{'Mass Mean (mu)':<28} | {'5279.83 MeV':<20} | {'5280.07 MeV':<20}")
print(f"{'Mass Resolution (sigma)':<28} | {'17.59 MeV':<20} | {'17.26 MeV':<20}")
print("-"*75)
print(f"{'SIGNAL AREA (True Events)':<28} | {bdt_sig_area:<20.1f} | {manual_sig_area:<20.1f}")
print(f"{'BACKGROUND AREA (Noise)':<28} | {bdt_bkg_area:<20.1f} | {manual_bkg_area:<20.1f}")
print(f"{'TOTAL AREA under fit':<28} | {bdt_total_area:<20.1f} | {manual_total_area:<20.1f}")
print("-"*75)
print(f"{'SAMPLE PURITY (%)':<28} | {bdt_purity:<19.2f}% | {manual_purity:<19.2f}%")
print("="*75)