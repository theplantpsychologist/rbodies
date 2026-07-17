import os
import csv
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, medfilt

try:
    import cinereader as cr
except ImportError:
    print("Warning: cinereader not installed. .cine extraction will fail.")

PX_PER_CM = 80

# ==========================================
# 1. CORE COMPUTER VISION (Shared Helpers)
# ==========================================

def get_binary_mask(frame_data, threshold_value):
    """Converts raw frame to 8-bit grayscale and applies a binary threshold."""
    if len(frame_data.shape) == 3:
        gray_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame_data

    frame_norm = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(frame_norm, threshold_value, 255, cv2.THRESH_BINARY)
    return gray_frame, binary_mask

def extract_trapezoid_features(binary_mask):
    """
    Finds the right edge using corner projection (max x-y and x+y),
    calculates the average tilt of the top edge and the perpendicular of the right edge, 
    and walks left.
    """
    y_coords, x_coords = np.where(binary_mask == 255)
    if len(y_coords) == 0:
        return None, None, None

    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)

    # 1. Find Top-Right and Bottom-Right Corners
    # Image origin (0,0) is top-left. +x is right, +y is down.
    # TR maximizes x, minimizes y -> maximize (x - y)
    # BR maximizes x, maximizes y -> maximize (x + y)
    tr_idx = np.argmax(x_coords - y_coords)
    br_idx = np.argmax(x_coords + y_coords)
    
    tr_pt = (x_coords[tr_idx], y_coords[tr_idx])
    br_pt = (x_coords[br_idx], y_coords[br_idx])

    # 2. Extract the right edge strictly between these two corners
    y_min_bound = min(tr_pt[1], br_pt[1])
    y_max_bound = max(tr_pt[1], br_pt[1])
    
    # Shave off 5% from the top and bottom to avoid rounding at the physical corners
    margin = int(max(1, (y_max_bound - y_min_bound) * 0.05))
    
    # Get raw rightmost pixels for every row
    raw_right_x = np.array([np.max(x_coords[y_coords == y]) for y in unique_y])
    
    # Filter for the vertical stretch
    valid_mask = (unique_y >= y_min_bound + margin) & (unique_y <= y_max_bound - margin)
    
    # Fallback if the shape is extremely small and the margin erased it
    if not np.any(valid_mask):
        valid_mask = (unique_y >= y_min_bound) & (unique_y <= y_max_bound)

    right_edge_x = raw_right_x[valid_mask]
    right_edge_y = unique_y[valid_mask]

    # 3. Find Midpoint (p1) using this cleanly defined edge
    y_mid_target = (np.min(right_edge_y) + np.max(right_edge_y)) / 2
    closest_idx = np.argmin(np.abs(right_edge_y - y_mid_target))
    p1 = (right_edge_x[closest_idx], right_edge_y[closest_idx])

    # 4. Fit the True Right Edge & Find its Perpendicular
    # Fit x as a function of y to prevent infinity crashes on vertical lines
    if len(right_edge_y) > 1:
        m_y, _ = np.polyfit(right_edge_y, right_edge_x, 1)
        slope_perp = -m_y 
    else:
        slope_perp = 0.0

    # 5. Fit the Top Edge
    # Use only the right half of the shape to avoid the left slanted edge
    # x_mid = (np.min(unique_x) + np.max(unique_x)) / 2
    # right_half_x = unique_x[unique_x > x_mid]
    # if len(right_half_x) < 2:
    #     right_half_x = unique_x 

    # top_edge_y = np.array([np.min(y_coords[x_coords == x]) for x in right_half_x])
    
    # if len(right_half_x) > 1:
    #     slope_top, _ = np.polyfit(right_half_x, top_edge_y, 1)
    # else:
    #     slope_top = 0.0

    # # 6. Average the angles
    # angle_top = np.arctan(slope_top)
    # angle_perp = np.arctan(slope_perp)
    # angle_avg = (angle_top + angle_perp) / 2.0
    # avg_slope = np.tan(angle_avg)

    # 7. Walk left (-x) along the perpendicular slope to find p2
    perp_angle = np.arctan(slope_perp)

    p2 = None
    for x in range(p1[0], -1, -1):
        y = int(round(p1[1] + perp_angle * (x - p1[0])))
        
        # Bounds check and collision detection
        if y < 0 or y >= binary_mask.shape[0] or binary_mask[y, x] == 0:
            p2 = (x, np.clip(y, 0, binary_mask.shape[0] - 1))
            break
            
    if p2 is None:
        p2 = (0, int(round(p1[1] + perp_angle * (0 - p1[0]))))

    # Package debug data
    debug_edges = {
        'right_x': right_edge_x, 'right_y': right_edge_y,
        'slope_perp': slope_perp,
        'tr_pt': tr_pt,
        'br_pt': br_pt
    }
    return p1, p2, debug_edges
# ==========================================
# 2. PROCESSING PIPELINES
# ==========================================

def compress_video(file_path, output_mp4=None, scale_factor=0.5):
    """
    Standalone pipeline to purely compress a video (.cine or .mp4) to a lightweight .mp4.
    Safely handles 16-bit to 8-bit conversion to prevent pixel wrapping.
    """
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    # Determine input file type
    file_ext = os.path.splitext(file_path)[1].lower()
    is_cine = (file_ext == '.cine')

    # Auto-generate output filename if not provided
    if output_mp4 is None:
        output_mp4 = file_path.rsplit('.', 1)[0] + '.mp4'

    print(f"Opening {file_ext.upper()} file...")

    # --- 1. INITIALIZATION & METADATA ---
    if is_cine:
        metadata = cr.read_metadata(file_path)
        total_frames = metadata.ImageCount
        orig_width = metadata.ImWidth
        orig_height = metadata.ImHeight
        first_idx = metadata.FirstImageNo
    else:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_idx = 0

    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    print(f"Found {total_frames} frames ({orig_width}x{orig_height}).")
    print(f"Compressing to {new_width}x{new_height}...")

    # --- 2. VIDEO WRITER SETUP ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_mp4, fourcc, 60.0, (new_width, new_height), isColor=False)

    # --- 3. COMPRESSION LOOP ---
    for i in range(total_frames):
        frame_data = None
        
        # Read Frame
        if is_cine:
            target_frame = first_idx + i
            try:
                _, images, _ = cr.read(file_path, target_frame, 1)
                if images and len(images) > 0: 
                    frame_data = images[0]
            except Exception:
                pass 
        else:
            ret, frame = cap.read()
            if ret:
                # MP4s read via OpenCV might be BGR, force to grayscale
                if len(frame.shape) == 3:
                    frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame_data = frame

        if frame_data is None:
            continue

        # --- THE BUG FIX: 8-Bit Normalization ---
        # Compresses the 16-bit raw array down to 8-bit so the VideoWriter doesn't choke and wrap the white pixels
        frame_8bit = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Resize and Write
        frame_resized = cv2.resize(frame_8bit, (new_width, new_height), interpolation=cv2.INTER_AREA)
        video_writer.write(frame_resized)

        if (i + 1) % 100 == 0: 
            print(f"Compressed {i + 1}/{total_frames} frames...")

    # --- 4. CLEANUP ---
    if not is_cine:
        cap.release()
    video_writer.release()
    
    print(f"\nCompression Complete!\nSaved to: {output_mp4}")

def process_video(file_path, r, output_csv = None,output_mp4=None, fps=1000, threshold=128, scale_factor=0.5):
    """
    Unified Pipeline: Extracts kinematics from either a .cine or .mp4 file.
    Applies denoising and phase unwrapping, and saves the PROCESSED data to CSV.
    """

    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    # Determine input file type
    file_ext = os.path.splitext(file_path)[1].lower()
    is_cine = (file_ext == '.cine')

    print(f"Opening {file_ext.upper()} file...")
    if output_csv is None:
        output_csv = file_path.rsplit('.', 1)[0] + '.csv'
    # --- 1. INITIALIZATION & METADATA ---
    if is_cine:
        metadata = cr.read_metadata(file_path)
        total_frames = metadata.ImageCount
        orig_width = metadata.ImWidth
        orig_height = metadata.ImHeight
        first_idx = metadata.FirstImageNo
    else:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_idx = 0

    print(f"Found {total_frames} frames ({orig_width}x{orig_height}). Beginning processing...")

    # --- 2. OPTIONAL VIDEO WRITER SETUP ---
    video_writer = None
    if is_cine:
        output_mp4 = file_path.rsplit('.', 1)[0] + '_reference.mp4' if output_mp4 is None else output_mp4
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_mp4, fourcc, 30.0, (new_width, new_height), isColor=False)
        print(f"Exporting compressed reference video at {new_width}x{new_height}")

    # --- 3. CORE PROCESSING LOOP (Gathering raw data into memory) ---
    raw_data_list = []

    for i in range(total_frames):
        time_s = i / fps
        frame_data = None
        
        # --- READ FRAME ---
        if is_cine:
            target_frame = first_idx + i
            try:
                _, images, _ = cr.read(file_path, target_frame, 1)
                if images and len(images) > 0: 
                    frame_data = images[0]
            except Exception:
                pass 
        else:
            ret, frame = cap.read()
            if ret:
                frame_data = frame

        if frame_data is None:
            raw_data_list.append([time_s, np.nan, np.nan])
            continue

        # --- CV EXTRACTION ---
        # gray_frame, binary_mask = get_binary_mask(frame_data, threshold)
        # p1, p2, _ = extract_trapezoid_features(binary_mask)
        
        # if p1 and p2:
        #     p1_x_cm = p1[0] / PX_PER_CM
        #     width_px = np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])
        #     width_cm = width_px / PX_PER_CM
        #     raw_data_list.append([time_s, p1_x_cm, width_cm])
        # else:
        #     raw_data_list.append([time_s, np.nan, np.nan])

        # --- VIDEO EXPORT ---
        if video_writer:
            frame_resized = cv2.resize(frame_data, (new_width, new_height), interpolation=cv2.INTER_AREA)
            video_writer.write(frame_resized)

        if (i + 1) % 100 == 0: 
            print(f"Processed {i + 1}/{total_frames} frames...")

    # Clean up video objects
    if not is_cine:
        cap.release()
    if video_writer:
        video_writer.release()
        print(f"Reference video saved to: {output_mp4}")

    # # --- 4. SIGNAL PROCESSING & CSV EXPORT ---
    # print("\nApplying denoising and phase unwrapping...")
    
    # # Load raw data into DataFrame
    # df = pd.DataFrame(raw_data_list, columns=['time_s', 'position_cm', 'theta_raw_cm'])

    # # Run the unwrapping and smoothing logic
    # t_data, pos_data, theta_data = process_signals(df, r)

    # # Save the strictly processed data to CSV
    # df_processed = pd.DataFrame({
    #     'time_s': t_data,
    #     'position_cm': pos_data,
    #     'theta_deg': theta_data
    # })
    
    # df_processed.to_csv(output_csv, index=False)
    # print(f"Processed Data saved to: {output_csv}\nPipeline Complete!")


def debug_mp4_tracking(file_path, output_video_path, threshold=128):
    """
    Steps through an MP4 file, visually confirming the computer vision logic,
    and saves the Matplotlib visualization as a debug video.
    """
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return

    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting debug video render")

    # Disable interactive mode since we are saving to a file, not viewing live
    plt.ioff() 
    fig = plt.figure(figsize=(10, 90/16))
    
    video_writer = None

    for i in range(0, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
            
        gray_frame, binary_mask = get_binary_mask(frame, threshold)
        p1, p2, edges = extract_trapezoid_features(binary_mask)

        if p1 is None: continue

        # --- Visualizer ---
        plt.clf()
        
        # Calculate elapsed time based on original 1000 FPS
        elapsed_time = i / 1000.0
        fig.suptitle(f'Time: {elapsed_time:.3f}s', fontsize=16, fontweight='bold')

        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(gray_frame, cmap='gray')
        ax1.set_title('Original video', fontsize=12)
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 1, 2)
        ax2.imshow(binary_mask, cmap='gray')
        ax2.set_title('Extracted position and angle', fontsize=12)

        # Plot edges and points
        ax2.plot(edges['right_x'], edges['right_y'], color='magenta', linewidth=2, label='Right Edge')
        ax2.plot(*edges['tr_pt'], 'mo', markersize=5)
        ax2.plot(*edges['br_pt'], 'mo', markersize=5)

        ax2.plot(*p1, 'bo', markersize=8, label='Position')
        ax2.plot(*p2, 'ro', markersize=8)
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=2, label='Angle measurement')
        
        ax2.legend(loc='lower left')
        ax2.axis('off')
        
        plt.tight_layout()
        
        # --- Convert Matplotlib Figure to OpenCV Frame ---
        # Force a draw so the renderer updates the buffer
        fig.canvas.draw()
        
        # Extract the RGBA buffer from the figure
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        
        # Drop the Alpha channel to make it standard RGB, then convert to BGR for OpenCV
        rgb_frame = rgba_buffer[:, :, :3]
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Initialize the VideoWriter on the first frame once we know the exact pixel dimensions
        if video_writer is None:
            height, width, _ = bgr_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Set output fps to 60 so the debug video plays smoothly
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 60.0, (width, height))
            
        video_writer.write(bgr_frame)

        if (i) % 100 == 0:
            print(f"Rendered {i}/{total_frames} frames...")

    cap.release()
    if video_writer:
        video_writer.release()
    plt.close(fig) # Free memory
    
    print(f"Debug video successfully saved to: {output_video_path}")
# ==========================================
# 3. MATH & FFT MODELS
# ==========================================
def damped_sinusoid(t, amplitude, decay, freq, phase, offset):
    """Model function for an exponentially decaying sinusoid with a y-offset."""
    return amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase) + offset
def compute_fft(time_array, signal_array):
    valid = ~np.isnan(signal_array)
    t, sig = time_array[valid], signal_array[valid]
    if len(sig) == 0: return np.array([]), np.array([])
    
    sig_detrended = sig - np.mean(sig)
    N, dt = len(t), np.mean(np.diff(t))
    
    fft_values = np.fft.rfft(sig_detrended)
    freqs = np.fft.rfftfreq(N, d=dt)
    return freqs, np.abs(fft_values) / N * 2 

# ==========================================
# 4. DATA ANALYSIS & PLOTTING
# ==========================================
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df = df[df['time_s'] >= 0.3].copy()
    df['time_s'] = df['time_s'] - df['time_s'].iloc[0]
    return df

def process_signals(df, r):
    # --- Position ---
    pos_centered = df['position_cm'] - df['position_cm'].mean()
    pos_smoothed = pos_centered.rolling(window=5, center=True).mean()
    
    # --- Theta (Predictive Kinematic Unwrapping) ---
    time_s = df['time_s'].values
    
    theta_raw_deg = (2 * df['theta_raw_cm'].values) / r * (180 / np.pi)
    theta_clean = medfilt(theta_raw_deg, kernel_size=3)
    
    unwrapped_theta = np.copy(theta_clean)
    jump_indices = np.where(np.abs(np.diff(theta_clean)) > 10)[0]
    
    for i in jump_indices:
        x1 = unwrapped_theta[i]
        x2 = unwrapped_theta[i+1]
        
        start_idx = max(0, i - 5)
        if i > start_idx:
            v1 = (unwrapped_theta[i] - unwrapped_theta[start_idx]) / (time_s[i] - time_s[start_idx])
        else:
            v1 = 0.0
            
        dt = time_s[i+1] - time_s[i]
        x2_prime = x1 + (v1 * dt)
        offset = x2_prime - x2
        unwrapped_theta[i+1:] += offset
        
    theta_series = pd.Series(unwrapped_theta)
    theta_final = theta_series.rolling(window=5, center=True).mean().values
    
    # Add a safety check in case the dataframe became entirely NaNs
    valid_data = theta_series.dropna()
    if not valid_data.empty:
        theta_final = theta_final - valid_data.iloc[-1] 

    return time_s, pos_smoothed.values, theta_final

def fit_position_decay(t_data, y_data):
    """Fits the data to a damped sinusoid, allowing for a constant vertical offset."""
    valid = ~np.isnan(y_data)
    t_clean, y_clean = t_data[valid], y_data[valid]
    if len(y_clean) < 10: return None, None

    guess_amplitude = (np.max(y_clean) - np.min(y_clean)) / 2
    guess_offset = np.mean(y_clean) 
    
    zero_crossings = np.where(np.diff(np.sign(y_clean - guess_offset)))[0]
    guess_freq = 1 / (2 * np.mean(np.diff(t_clean[zero_crossings]))) if len(zero_crossings) > 1 else 2.0
        
    try:
        p0 = [guess_amplitude, 1.0, guess_freq, 0.0, guess_offset]
        popt, _ = curve_fit(damped_sinusoid, t_clean, y_clean, p0=p0)
        return popt, f'Fit: decay={popt[1]:.3f}, freq={abs(popt[2]):.2f}Hz'
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None
    

def plot_combined_dashboard(t_data, pos_data, theta_data, r, popt_pos, popt_theta, fit_label_pos, fit_label_theta, title, do_fft=True):
    """
    Plots a 2x2 dashboard.
    Left Column: Time-domain decay (Position and Theta)
    Right Column: Frequency-domain FFT (Position and Theta)
    """
    # Create a 2x2 grid. 
    # sharex='col' ensures the left column shares the Time axis, and the right shares the Freq axis.
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex='col')
    
    # Unpack axes for readability
    ax_pos_time = axs[0, 0]
    ax_theta_time = axs[1, 0]
    ax_pos_fft = axs[0, 1]
    ax_theta_fft = axs[1, 1]

    # ==========================================
    # LEFT COLUMN: TIME DOMAIN
    # ==========================================
    
    # --- Top Left: Position ---
    ax_pos_time.plot(t_data, pos_data, color='b', linewidth=1.5, label='Position')
    if popt_pos is not None: 
        ax_pos_time.plot(t_data, damped_sinusoid(t_data, *popt_pos), 'k--', linewidth=2, label=fit_label_pos)
        
    ax_pos_time.set_ylabel('Position (cm)', fontweight='bold')
    ax_pos_time.set_title(f"{title} - Time Domain", fontsize=14, fontweight='bold')
    ax_pos_time.grid(True, linestyle='--', alpha=0.7)
    ax_pos_time.legend(loc='upper right')

    # --- Bottom Left: Theta ---
    ax_theta_time.plot(t_data, theta_data, color='r', linewidth=1.5, label='Theta Angle (deg)')
    if popt_theta is not None:
        ax_theta_time.plot(t_data, damped_sinusoid(t_data, *popt_theta), 'k--', linewidth=2, label=fit_label_theta)
        
    ax_theta_time.set_xlabel('Time (s)', fontweight='bold')
    ax_theta_time.set_ylabel('Theta Angle (deg)', fontweight='bold')
    ax_theta_time.grid(True, linestyle='--', alpha=0.7)
    ax_theta_time.legend(loc='upper right')

    # ==========================================
    # RIGHT COLUMN: FREQUENCY DOMAIN (FFT)
    # ==========================================
    if do_fft:
        max_freq = 20.0
        
        # Calculate FFTs
        f_pos, mag_pos = compute_fft(t_data, pos_data)
        f_theta, mag_theta = compute_fft(t_data, theta_data)

        # Loop to apply styling and peak labeling symmetrically
        fft_data = [
            (ax_pos_fft, f_pos, mag_pos, 'b', 'Position FFT'),
            (ax_theta_fft, f_theta, mag_theta, 'r', 'Theta FFT')
        ]

        for ax, f, mag, color, subtitle in fft_data:
            if len(f) > 0 and len(mag) > 0:
                ax.plot(f, mag, color=color, linewidth=1.5)
                peaks, _ = find_peaks(mag, height=1.0)
                
                # Label peaks
                for p in peaks:
                    if f[p] <= max_freq:
                        ax.plot(f[p], mag[p], f"x{color}")
                        ax.annotate(f'{f[p]:.2f} Hz\n({mag[p]:.1f})', xy=(f[p], mag[p]), 
                                    xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax.set_xlim(0, max_freq)
            ax.set_ylabel('Magnitude', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)

        ax_pos_fft.set_title(f"{title} - Frequency Domain", fontsize=14, fontweight='bold')
        ax_theta_fft.set_xlabel('Frequency (Hz)', fontweight='bold')
    else:
        # Hide the right column entirely if FFT is toggled off
        ax_pos_fft.set_visible(False)
        ax_theta_fft.set_visible(False)

    plt.tight_layout()
    # plt.show()
    save_path = f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(save_path, dpi=300)
    print(f"Dashboard saved as '{save_path}'")

def analyze_kinematic_data(csv_file_path, r, title="", do_fft=True):
    """
    Loads ALREADY PROCESSED data, fits the curves, and generates the dashboard.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return

    # Load the explicitly processed columns from the CSV
    df = pd.read_csv(csv_file_path)
    t_data = df['time_s'].values
    pos_data = df['position_cm'].values
    theta_data = df['theta_deg'].values
    
    popt_pos, fit_label_pos = fit_position_decay(t_data, pos_data)
    popt_theta, fit_label_theta = fit_position_decay(t_data, theta_data)
    
    plot_combined_dashboard(
        t_data, pos_data, theta_data, r, 
        popt_pos, popt_theta, 
        fit_label_pos, fit_label_theta, 
        title=title, 
        do_fft=do_fft
    )
# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    circ = 8.5
    # name = f"sample_a_circ_{circ}_linearresonance"

    # video_path = f"/Users/bwong/Desktop/rbodies/dynamics/videos/{name}.mp4"
    # csv_path = f"/Users/bwong/Desktop/rbodies/dynamics/processed_data/{name}.csv"
    # video_path = f"/Volumes/maxone/Brandon/double.cine"
    # debug_video_path = f"/Users/bwong/Desktop/rbodies/dynamics/videos/debug_sample_b_circ_7_unwound_linear_resonance.mp4"
    # debug_mp4_tracking(video_path, debug_video_path, threshold=130)

    # process_video("/Volumes/maxone/Brandon/torsion_resonance.cine", output_csv=None, fps=1000, threshold=130, scale_factor=0.5, r=circ/(2*np.pi))
    compress_video("/Volumes/maxone/Brandon/double.cine")
    compress_video("/Volumes/maxone/Brandon/locking3.cine")
    compress_video("/Volumes/maxone/Brandon/torsion_resonance.cine")


    # compress_video("/Volumes/maxone/Brandon/freq_sweep.cine")

    
    
    # analyze_kinematic_data(csv_path, r=circ/(2*np.pi), title = "10cm Linear Resonance", )

"""
sample a linear resonance: paper triangle of dimensions 3x6 cm

"""
