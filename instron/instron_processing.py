import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def moving_average(data, window_size=30):
    # Create uniform weights that sum up to 1
    window = np.ones(window_size) / window_size
    
    # 'valid' mode avoids edge effects but shortens the output array
    return np.convolve(data, window, mode='valid')

def plot_instron_data(file_path):
    # Let it crash naturally here if the path or file is bad
    df = pd.read_csv(file_path)
    
    # Strip whitespace from columns to avoid indexing errors
    df.columns = df.columns.str.strip()
    
    # Extract raw data by position (assuming displacement is 2nd, force is 3rd)
    displacement_mm = df.iloc[1:, 1].astype(float)  # Convert to float for plotting
    force_kN = df.iloc[1:, 2].astype(float)  # Convert to float for plotting
    
    # Unit conversions: mm -> m and kN -> N
    displacement_m = displacement_mm / 1000.0
    force_N = force_kN * 1000.0
    
    # Plotting configuration
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Style: Thinner line or slight alpha can help expose overlapping data points
    ax.plot(displacement_m, force_N, color='royalblue', linewidth=1.5, alpha=0.9)
    
    ax.set_title('Force vs. Displacement')
    ax.set_xlabel('Displacement (m)')
    ax.set_ylabel('Force (N)')
    
    # Fix the overcrowded axis ticks
    # MaxNLocator guarantees no more than N neatly spaced intervals
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    
    # Clean up display formatting (e.g., standard decimal notation instead of scientific)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_multiple_runs(file_paths, labels=None):
    """
    Plots Force vs. Displacement for multiple Instron CSV runs on a single axis.
    
    file_paths: list of strings, e.g., ['run1.csv', 'run2.csv']
    labels: list of strings for the legend, e.g., ['0.5mm/min', '1.0mm/min']
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Track runs to handle legend labels if none are provided
    for idx, path in enumerate(file_paths):
        df = pd.read_csv(path)
    
        # Strip whitespace from columns to avoid indexing errors
        df.columns = df.columns.str.strip()
        
        # Extract raw data by position (assuming displacement is 2nd, force is 3rd)
        displacement_mm = df.iloc[1:, 1].astype(float)  # Convert to float for plotting
        force_kN = df.iloc[1:, 2].astype(float)  # Convert to float for plotting
        
        # Unit conversions: mm -> m and kN -> N
        displacement_m = displacement_mm / 1000.0
        force_N = force_kN * 1000.0
                
        # Style: Thinner line or slight alpha can help expose overlapping data points
        ax.semilogy(moving_average(displacement_m), moving_average(force_N), label=labels[idx] if labels else f"Run {idx + 1}")

    # Global plot styling
    ax.set_title('Force vs. Displacement - Combined Test Runs', fontsize=14, pad=15)
    ax.set_xlabel('Displacement (m)', fontsize=11)
    ax.set_ylabel('Force (N)', fontsize=11)
    
    # Control axis tick density
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # plot_multiple_runs([
    #     "sorted_data/yoyo-hysteresis_1.csv",
    #     "sorted_data/yoyo-hysteresis-0wind-after-manytrials_1.csv"
    # ])
    plot_multiple_runs([
        "sorted_data/3yoyo-hysteresis-10winds__1.csv",
        "sorted_data/5yoyo-hysteresis-10winds-trial2_1.csv",
        "sorted_data/6yoyo-hysteresis-10wind-trial3_1.csv",
        "sorted_data/7yoyo-hysteresis-10wind-trial4_1.csv",
        "sorted_data/4yoyo-hysteresis-5winds_1.csv"
    ], labels=[
        "Trial 1 - 10 Winds",
        "Trial 2 - 10 Winds",
        "Trial 3 - 10 Winds",
        "Trial 4 - 10 Winds",
        "Trial 1 - 5 Winds"
    ])

    # plot_multiple_runs([
    #     "sorted_data/9yoyo-hysteresis-sample2-wind0-trial1_1.csv",
    #     "sorted_data/10yoyo-hysteresis-sample2-windNeg5-trial1_1.csv",
    #     "sorted_data/11yoyo-hysteresis-sample2-windNeg10-trial1_1.csv",
    # ], labels=[
    #     "Trial 1 - No Wind",
    #     "Trial 1 - -5 Winds",
    #     "Trial 1 - -10 Winds",
    # ])


   