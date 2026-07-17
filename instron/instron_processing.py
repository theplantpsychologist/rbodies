import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
# from skimage.restoration import denoise_tv_chambolle
# from scipy.signal import savgol_filter

def log_double_exponential(d, log_A, m1, log_B, m2):
    A = np.exp(log_A)
    B = np.exp(log_B)
    return np.log(A * np.exp(m1 * d) + B * np.exp(m2 * d))

def adaptive_log_smooth(displacement, force, min_window=3):
    """
    Smoothes log-force with a window size inversely proportional to the local force:
    Window Size = ceil(10 / Force).
    
    min_window: The minimum odd window size to prevent over-smoothing 
                when force gets exceptionally high. Set to 1 if you want 
                literally zero smoothing at higher forces.
    """
    log_force = np.log(force)
    n_points = len(log_force)
    smoothed_log = np.zeros(n_points)
    
    for i in range(n_points):
        # Calculate local window size based on physical Force in Newtons
        f_local = force[i]
        
        # Avoid division by zero if any force is exactly 0
        if f_local <= 0:
            local_window = 1001  # Fallback to large window for near-zero values
        else:
            # 10 / force calculation, rounded up to the nearest odd integer
            calculated_w = int(np.ceil(100.0 / f_local))
            # Enforce minimum window limit (and ensure it is odd)
            local_window = max(calculated_w, min_window)
            
        if local_window % 2 == 0:
            local_window += 1
            
        # Define window boundaries centered around index i
        half_w = local_window // 2
        start_idx = max(0, i - half_w)
        end_idx = min(n_points, i + half_w + 1)
        
        # Compute mean in the log domain
        smoothed_log[i] = np.mean(log_force[start_idx:end_idx])
        
    return np.exp(smoothed_log)

def to_sig_figs(num, sig_figs=2):
    if num == 0:
        return "0"
    # Calculate order of magnitude
    order = int(np.floor(np.log10(abs(num))))
    # Round to the target number of significant figures
    val = round(num, -order + (sig_figs - 1))
    
    # Format scientific notation elegantly for LaTeX if values are tiny/huge
    if abs(val) < 0.01 or abs(val) >= 10000:
        factor = val / (10**order)
        # Handle rounding changes to order
        if abs(factor) >= 10:
            factor /= 10
            order += 1
        return f"{factor:.1f} \\times 10^{{{order}}}"
    else:
        # Standard decimal presentation
        decimals = max(0, -order + (sig_figs - 1))
        return f"{val:.{decimals}f}"

def plot_multiple_runs(file_paths, labels=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use a standard qualitative colormap to assign unique matching colors per run
    colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))
    
    for idx, path in enumerate(file_paths):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        
        # 1. Load and parse raw values
        displacement_mm = df.iloc[1:, 1].astype(float).reset_index(drop=True)
        force_kN = df.iloc[1:, 2].astype(float).reset_index(drop=True)
        
        d_raw = (displacement_mm / 1000.0).values
        f_raw = (force_kN * 1000.0).values
        
        # 2. Slice at peak force
        peak_idx = f_raw.argmax()
        d_raw = d_raw[:peak_idx + 1]
        f_raw = f_raw[:peak_idx + 1]
        
        # 3. Discard forces lower than 0.1 N to preserve log domain fitting
        valid_mask = f_raw >= 0.1
        d_raw = d_raw[valid_mask]
        f_raw = f_raw[valid_mask]
        
        if len(f_raw) < 10:
            continue
            
        color = colors[idx]
        run_name = labels[idx] if labels and idx < len(labels) else f"Run {idx + 1}"
        
        # 4. Perform Curve Fit on RAW, UN-SMOOTHED data in log space
        log_f_raw = np.log(f_raw)
        
        # Smart initial guess:
        # log_A and log_B set near low/high force states, m1 is a slow rate, m2 is a fast rate.
        initial_guess = [np.log(0.1), 10.0, np.log(1.0), 100.0]
        
        try:
            # Let it raise an error to terminal if curve_fit completely fails to converge
            popt, pcov = curve_fit(log_double_exponential, d_raw, log_f_raw, p0=initial_guess, maxfev=10000)
            
            # Convert fitting parameters back to linear domain values
            A = np.exp(popt[0])
            m1 = popt[1]
            B = np.exp(popt[2])
            m2 = popt[3]
            
            # Format parameters with exactly 2 significant figures
            A_str = to_sig_figs(A)
            m1_str = to_sig_figs(m1)
            B_str = to_sig_figs(B)
            m2_str = to_sig_figs(m2)
            
            # Formulate the mathematical LaTeX label for the legend
            legend_label = f"{run_name}: $F(d) = {A_str} e^{{{m1_str} d}} + {B_str} e^{{{m2_str} d}}$"
            
        except RuntimeError:
            print(f"Warning: Fit failed to converge for {run_name}. Plotting without fit.")
            legend_label = f"{run_name} (Fit Failed)"
            A = m1 = B = m2 = None

        # 5. Smooth the data for visualization using our adaptive log smoothing
        f_smoothed = adaptive_log_smooth(d_raw, f_raw, min_window=3)
        
        # 6. Plotting elements
        # Faded gray background for raw staircase data (unlabeled)
        ax.semilogy(d_raw, f_raw, color='gray', alpha=0.15, drawstyle='steps-post')
        
        # Solid colored line for adaptive smoothed data (this one gets the main labeled legend)
        ax.semilogy(d_raw, f_smoothed, color=color, linewidth=2.0, label=legend_label)
        
        # Dashed colored line for the theoretical curve fit (matching the solid line's color)
        if A is not None:
            f_fit = A * np.exp(m1 * d_raw) + B * np.exp(m2 * d_raw)
            ax.semilogy(d_raw, f_fit, color=color, linestyle='--', linewidth=1.5, alpha=0.9)

    # Clean styling
    ax.set_title('Double Exponential Curve Fit of Force vs. Displacement', fontsize=14, pad=15)
    ax.set_xlabel('Displacement $d$ (m)', fontsize=11)
    ax.set_ylabel('Force $F$ (N)', fontsize=11)
    
    # Axis tick control
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend(loc='best', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # plot all
    plot_multiple_runs([
        "instron/sorted_data/1yoyo1_1.csv",
        "instron/sorted_data/2yoyo-hysteresis_1.csv",
        "instron/sorted_data/3yoyo-hysteresis-10winds__1.csv",
        "instron/sorted_data/4yoyo-hysteresis-5winds_1.csv",
        "instron/sorted_data/5yoyo-hysteresis-10winds-trial2_1.csv",
        "instron/sorted_data/6yoyo-hysteresis-10wind-trial3_1.csv",
        "instron/sorted_data/7yoyo-hysteresis-10wind-trial4_1.csv",
        "instron/sorted_data/8yoyo-hysteresis-0wind-after-manytrials_1.csv",
        "instron/sorted_data/9yoyo-hysteresis-sample2-wind0-trial1_1.csv",
        "instron/sorted_data/10yoyo-hysteresis-sample2-windNeg5-trial1_1.csv",
        "instron/sorted_data/11yoyo-hysteresis-sample2-windNeg10-trial1_1.csv"
    ], labels=[
        
    ])

    # Plot specific selection
    plot_multiple_runs([
        # "instron/sorted_data/1yoyo1_1.csv",
        "instron/sorted_data/2yoyo-hysteresis_1.csv",
        "instron/sorted_data/4yoyo-hysteresis-5winds_1.csv",
        "instron/sorted_data/3yoyo-hysteresis-10winds__1.csv",
        # "instron/sorted_data/5yoyo-hysteresis-10winds-trial2_1.csv",
        # "instron/sorted_data/6yoyo-hysteresis-10wind-trial3_1.csv",
        # "instron/sorted_data/7yoyo-hysteresis-10wind-trial4_1.csv",
        # "instron/sorted_data/8yoyo-hysteresis-0wind-after-manytrials_1.csv",
        # "instron/sorted_data/9yoyo-hysteresis-sample2-wind0-trial1_1.csv",
        # "instron/sorted_data/10yoyo-hysteresis-sample2-windNeg5-trial1_1.csv",
        # "instron/sorted_data/11yoyo-hysteresis-sample2-windNeg10-trial1_1.csv"
    ], labels=[
        "Run 2 (0 pre-winds)",
        "Run 4 (5 pre-winds)",
        "Run 3 (10 pre-winds)"
    ])

   