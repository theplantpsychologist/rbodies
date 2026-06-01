import os
import csv
import numpy as np
import cv2
import cinereader as cr
import pandas as pd
import matplotlib.pyplot as plt

PX_PER_CM = 80

def analyze_trapezoid_silent(frame_data, threshold_value=128):
    """
    Identical to the previous function, but completely silent (no plots).
    Returns p1 and p2.
    """
    frame_norm = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(frame_norm, threshold_value, 255, cv2.THRESH_BINARY)

    y_coords, x_coords = np.where(binary_mask == 255)
    
    if len(y_coords) == 0:
        return None, None

    unique_y = np.unique(y_coords)
    right_edge_x = []
    
    for y in unique_y:
        rightmost_x = np.max(x_coords[y_coords == y])
        right_edge_x.append(rightmost_x)

    y_min = np.min(unique_y)
    y_max = np.max(unique_y)
    y_mid = int((y_min + y_max) / 2)

    closest_idx = np.argmin(np.abs(unique_y - y_mid))
    p1_y = unique_y[closest_idx]
    p1_x = right_edge_x[closest_idx]
    p1 = (p1_x, p1_y)

    p2 = None
    for x in range(p1_x, -1, -1):
        if binary_mask[p1_y, x] == 0:
            p2 = (x, p1_y)
            break
            
    if p2 is None:
        p2 = (0, p1_y)

    return p1, p2

# def process_cine_to_csv(file_path, output_csv, fps=1000, threshold=128):
#     """
#     Processes all frames in a .cine file and writes the extraction data to a CSV.
#     Includes error handling for corrupted frames.
#     """
#     if not os.path.exists(file_path):
#         print(f"Error: The file '{file_path}' does not exist.")
#         return

#     print(f"Reading metadata for {file_path}...")
#     metadata = cr.read_metadata(file_path)
#     first_idx = metadata.FirstImageNo
#     total_frames = metadata.ImageCount

#     print(f"Found {total_frames} frames. Beginning processing...")

#     with open(output_csv, mode='w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(['time_s', 'position_cm', 'theta_raw_cm'])

#         for i in range(total_frames):
#             target_frame = first_idx + i
#             time_s = i / fps
            
#             # --- FAULT TOLERANT READ ---
#             try:
#                 _, images, _ = cr.read(file_path, target_frame, 1)
#             except ValueError as ve:
#                 print(f"Warning: Corrupted metadata at frame {target_frame} ({ve}). Skipping.")
#                 writer.writerow([time_s, 'NaN', 'NaN'])
#                 continue
#             except Exception as e:
#                 print(f"Warning: Unexpected read error at frame {target_frame} ({e}). Skipping.")
#                 writer.writerow([time_s, 'NaN', 'NaN'])
#                 continue
#             # ---------------------------
            
#             if not images or len(images) == 0:
#                 print(f"Warning: No image data returned for frame {target_frame}. Skipping.")
#                 writer.writerow([time_s, 'NaN', 'NaN'])
#                 continue
                
#             frame_data = images[0]
            
#             # Extract points
#             p1, p2 = analyze_trapezoid_silent(frame_data, threshold_value=threshold)
            
#             if p1 and p2:
#                 p1_x = p1[0]
#                 diff_x = p1[0] - p2[0]
#                 p1_x_cm = p1_x / PX_PER_CM
#                 diff_x_cm = diff_x / PX_PER_CM
#                 writer.writerow([time_s, p1_x_cm, diff_x_cm])
#             else:
#                 writer.writerow([time_s, 'NaN', 'NaN'])

#             # Progress update
#             if (i + 1) % 100 == 0 or (i + 1) == total_frames:
#                 print(f"Processed {i + 1}/{total_frames} frames...")

#     print(f"\nDone! Data saved successfully to: {output_csv}")


def process_and_compress_cine(file_path, output_csv, output_mp4, fps=1000, threshold=128, scale_factor=0.5):
    """
    1-Pass Pipeline: Extracts kinematic data to CSV while simultaneously 
    compressing the raw .cine frames into a lightweight reference MP4.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    print(f"Reading metadata for {file_path}...")
    metadata = cr.read_metadata(file_path)
    first_idx = metadata.FirstImageNo
    total_frames = metadata.ImageCount
    
    # Get original resolution
    orig_width = metadata.ImWidth
    orig_height = metadata.ImHeight
    
    # Calculate new compressed resolution
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    print(f"Original Resolution: {orig_width}x{orig_height}")
    print(f"Compressed Resolution: {new_width}x{new_height}")
    print(f"Found {total_frames} frames. Beginning 1-pass processing...")

    # Initialize the VideoWriter
    # 'mp4v' is the standard, globally supported OpenCV codec for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # We set playback FPS to 30 so you can actually watch it in slow-motion.
    # We set isColor=False because we are feeding it an 8-bit grayscale array.
    video_writer = cv2.VideoWriter(output_mp4, fourcc, 30.0, (new_width, new_height), isColor=False)

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['time_s', 'position_cm', 'theta_raw_cm'])

        for i in range(total_frames):
            target_frame = first_idx + i
            time_s = i / fps
            
            # --- FAULT TOLERANT READ ---
            try:
                _, images, _ = cr.read(file_path, target_frame, 1)
            except Exception as e:
                print(f"Warning: Read error at frame {target_frame} ({e}). Skipping.")
                csv_writer.writerow([time_s, 'NaN', 'NaN'])
                continue
            
            if not images or len(images) == 0:
                csv_writer.writerow([time_s, 'NaN', 'NaN'])
                continue
                
            frame_data = images[0]
            
            # --- 1. DATA EXTRACTION (Math) ---
            p1, p2 = analyze_trapezoid_silent(frame_data, threshold_value=threshold)
            
            if p1 and p2:
                p1_x = p1[0]
                diff_x = p1[0] - p2[0]
                p1_x_cm = p1_x / PX_PER_CM
                diff_x_cm = diff_x / PX_PER_CM
                csv_writer.writerow([time_s, p1_x_cm, diff_x_cm])
            else:
                csv_writer.writerow([time_s, 'NaN', 'NaN'])

            # --- 2. VIDEO COMPRESSION (Visual) ---
            # Step A: Normalize 12/14-bit raw sensor data down to 8-bit (0-255)
            frame_8bit = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Step B: Scale down the resolution aggressively
            frame_resized = cv2.resize(frame_8bit, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Step C: Write the frame to the compressed MP4
            video_writer.write(frame_resized)

            # Progress update
            if (i + 1) % 100 == 0 or (i + 1) == total_frames:
                print(f"Processed {i + 1}/{total_frames} frames...")

    # CRITICAL: Release the video writer to finalize the MP4 file, or it will be unplayable
    video_writer.release()
    
    print(f"\nPipeline Complete!")
    print(f"Data saved to: {output_csv}")
    print(f"Video saved to: {output_mp4}")


def plot_kinematic_data(csv_file_path):
    """
    Reads the exported CSV and plots position and theta_raw against time.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: Could not find the file '{csv_file_path}'.")
        return

    print(f"Loading data from {csv_file_path}...")
    
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Verify the columns exist to prevent cryptic errors
    expected_columns = ['time_s', 'position_cm', 'theta_raw_cm']
    for col in expected_columns:
        if col not in df.columns:
            print(f"Error: Expected column '{col}' not found in the CSV.")
            return

    # Create a figure with 2 vertically stacked subplots that share the X axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top Plot: Position vs Time ---
    ax1.plot(df['time_s'], df['position_cm'], color='b', linewidth=1.5, label='Position')
    ax1.set_ylabel('Position (cm)', fontweight='bold')
    ax1.set_title('Kinematics over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # --- Bottom Plot: Theta Raw vs Time ---
    ax2.plot(df['time_s'], df['theta_raw_cm'], color='r', linewidth=1.5, label='Theta Raw')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Theta Raw (cm)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')

    # Automatically adjust padding so labels don't overlap
    plt.tight_layout()
    
    # Display the plot
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    name = "sample_a_circ_7.5_linearresonance"
    # name = "sample"
    video_path = f"/Volumes/maxone/Brandon/sample_a_circ_7.5_linearresonance.cine"
    # video_path = f"dynamics/raw_videos/{name}.cine"
    output_csv = f"dynamics/processed_data/{name}.csv"
    output_mp4 = f"dynamics/processed_data/{name}.mp4"

    # You can adjust the FPS and threshold here
    process_and_compress_cine(video_path, output_csv, output_mp4, fps=1000, threshold=128)
    plot_kinematic_data(output_csv)

"""
sample a linear resonance: paper triangle of dimensions 3x6 cm

"""
