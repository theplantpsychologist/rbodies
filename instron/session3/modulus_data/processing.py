import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Update this pattern to match your file directory/naming convention
csv_files = glob.glob("*.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = ["red","red","grey","grey","green","green"]
for file in csv_files:
    label = os.path.splitext(os.path.basename(file))[0]

    # Skip row 1 (the units row: "(s),(mm),(kN)")
    df = pd.read_csv(file, skiprows=[1])

    # Ensure numeric conversion
    disp = pd.to_numeric(df["Displacement"], errors="coerce").to_numpy()
    force = pd.to_numeric(df["Force"], errors="coerce").to_numpy()

    # Drop any NaNs
    valid = ~(np.isnan(disp) | np.isnan(force))
    disp, force = disp[valid], force[valid]

    if len(disp) == 0:
        continue

    # Normalize curves to start strictly at (0, 0)
    disp_norm = disp - disp[0]
    force_norm = force - force[0]

    # Calculate stiffness (secant stiffness: F / d)
    # Avoid zero-division at the origin: ignore points where displacement == 0
    non_zero = disp_norm > 1e-9
    disp_stiff = disp_norm[non_zero]
    stiffness = force_norm[non_zero] / disp_stiff  # kN/mm

    # 1. Force vs. Displacement
    ax1.plot(disp_norm, force_norm, label=label,color = colors[int(label[0])-1])

    # 2. Stiffness vs. Displacement
    ax2.plot(disp_stiff[30:], stiffness[30:],color = colors[int(label[0])-1])

# Configure Force vs Displacement plot
ax1.set_title("Force vs. Displacement (Normalized)")
ax1.set_xlabel("Displacement (mm)")
ax1.set_ylabel("Force (kN)")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Configure Stiffness vs Displacement plot
ax2.set_title("Secant Stiffness ($F / \\Delta d$) vs. Displacement")
ax2.set_xlabel("Displacement (mm)")
ax2.set_ylabel("Stiffness (kN/mm)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()
print("done")