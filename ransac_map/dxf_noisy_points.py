import numpy as np
import matplotlib.pyplot as plt
import ezdxf

# Define variables
file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\vector_fitting\boot.dxf"
num_points_per_line = 25
noise_std = 0.01

# Function to load vectors from a DXF file
def load_lines_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    lines = [entity for entity in msp if entity.dxftype() == 'LINE']
    return lines

# Function to generate noisy points along vectors
def generate_noisy_points(lines, num_points=num_points_per_line, noise_std=noise_std):
    all_points = []
    for line in lines:
        start, end = np.array(line.dxf.start)[:2], np.array(line.dxf.end)[:2]  # Taking only x and y coordinates
        points_along_edge = np.linspace(start, end, num_points)
        noise = np.random.normal(scale=noise_std, size=(num_points, 2))
        noisy_points = points_along_edge + noise
        all_points.extend(noisy_points)
    return np.array(all_points)

# Function to visualize the results
def visualize_results(lines, noisy_points):
    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)

    for line in lines:
        plt.plot([line.dxf.start[0], line.dxf.end[0]], [line.dxf.start[1], line.dxf.end[1]], 'g-', label="Original Line")

    # Calculate the extents of the DXF data
    all_x_coords = [coord for line in lines for coord in [line.dxf.start[0], line.dxf.end[0]]]
    all_y_coords = [coord for line in lines for coord in [line.dxf.start[1], line.dxf.end[1]]]

    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)

    # Calculate padding (20%)
    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2

    # Set axis limits with padding
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)

    # Move legend outside the plot (upper-left)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Connected Vectors from DXF File')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    lines = load_lines_from_dxf(file_path)
    noisy_points = generate_noisy_points(lines)
    
    visualize_results(lines, noisy_points)
