import ezdxf
import matplotlib.pyplot as plt

def plot_connected_vectors(dxf_file_path):
    # Create a new figure
    fig, ax = plt.subplots()

    # Load the DXF file
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()

    # Initialize bounding box coordinates
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    # Iterate through the DXF entities and extract connected vectors
    for entity in msp.query('LINE'):
        points = [(entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y)]
        
        # Update bounding box coordinates
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        
        # Plot the vector as a line segment
        x_coords, y_coords = zip(*points)
        plt.plot(x_coords, y_coords, marker='o', linestyle='-')

    # Set axis limits based on the calculated bounding box
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)

    # Set axis labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Connected Vectors from DXF File')

    # Display the plot
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

# Usage example
if __name__ == "__main__":
    dxf_file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\bay_lines.dxf"
    plot_connected_vectors(dxf_file_path)
