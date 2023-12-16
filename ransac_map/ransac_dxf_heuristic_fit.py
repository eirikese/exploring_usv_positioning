import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from sklearn.linear_model import RANSACRegressor

# Define variables
file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\vector_fitting\boot.dxf"
num_points_per_line = 25
noise_std = 0.01

def load_lines_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    lines = [entity for entity in msp if entity.dxftype() == 'LINE']
    return lines

def generate_noisy_points(lines, num_points=num_points_per_line, noise_std=noise_std):
    all_points = []
    for line in lines:
        start, end = np.array(line.dxf.start)[:2], np.array(line.dxf.end)[:2]  # Taking only x and y coordinates
        points_along_edge = np.linspace(start, end, num_points)
        noise = np.random.normal(scale=noise_std, size=(num_points, 2))
        noisy_points = points_along_edge + noise
        all_points.extend(noisy_points)
    return np.array(all_points)

def fit_line_ransac(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac = RANSACRegressor().fit(X, y)
    inliers = np.array(ransac.inlier_mask_)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_, inliers

def find_intersection(m1, b1, m2, b2):
    if m1 == m2:  # Lines are parallel
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def calculate_extents(lines):
    all_x_coords = [coord for line in lines for coord in [line.dxf.start[0], line.dxf.end[0]]]
    all_y_coords = [coord for line in lines for coord in [line.dxf.start[1], line.dxf.end[1]]]

    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)

    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    return (x_min, x_max, x_padding), (y_min, y_max, y_padding)

def compute_intersections(fitted_lines, extents):
    (x_min, x_max, x_padding), (y_min, y_max, y_padding) = extents
    intersections = []

    for i, (m1, b1) in enumerate(fitted_lines):
        for j, (m2, b2) in enumerate(fitted_lines):
            if i != j:
                intersection = find_intersection(m1, b1, m2, b2)
                if intersection:
                    x, y = intersection
                    if (x_min - 1.3*x_padding <= x <= x_max + 1.3*x_padding) and (y_min - 1.3*y_padding <= y <= y_max + 1.3*y_padding):
                        intersections.append(intersection)

    return intersections

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_filtered_vectors_from_intersections(fitted_lines, intersections, noisy_points, min_distance=0.1):
    filtered_vectors = []

    for m, b in fitted_lines:
        points_on_line = []
        for intersection in intersections:
            x, y = intersection
            if abs(y - (m * x + b)) < 1e-5:  # Check if the intersection point lies on the line
                points_on_line.append((x, y))

        # Sort the points by the X-coordinate to find consecutive points on the same line
        sorted_points = sorted(points_on_line, key=lambda point: point[0])

        # Create vectors between consecutive intersection points on the line
        for i in range(len(sorted_points) - 1):
            start_point = sorted_points[i]
            end_point = sorted_points[i + 1]

            # Check for noisy points along the vector
            vector_length = distance(start_point, end_point)
            num_steps = int(vector_length / min_distance)

            vector_valid = True
            for step in range(1, num_steps):
                step_ratio = step / num_steps
                interpolated_point = (start_point[0] + step_ratio * (end_point[0] - start_point[0]),
                                       start_point[1] + step_ratio * (end_point[1] - start_point[1]))

                # Check if there are noisy points within the minimum distance
                min_dist = min(distance(point, interpolated_point) for point in noisy_points)
                if min_dist > min_distance:
                    vector_valid = False
                    break

            if vector_valid:
                filtered_vectors.append((start_point, end_point))

    return filtered_vectors

def merge_vectors(vectors):
    merged_vectors = []
    i = 0

    while i < len(vectors) - 1:
        vector1 = vectors[i]
        vector2 = vectors[i + 1]

        # Check if the vectors are connected (end of vector1 == start of vector2)
        if vector1[1] == vector2[0]:
            # Calculate the lengths of the vectors
            length1 = distance(vector1[0], vector1[1])
            length2 = distance(vector2[0], vector2[1])

            # Check if vector2 lengthens vector1
            if length2 > length1:
                # Merge the vectors by replacing the end of vector1 with the end of vector2
                merged_vector = (vector1[0], vector2[1])
                merged_vectors.append(merged_vector)
                i += 2  # Skip the next vector since it's no longer needed
            else:
                merged_vectors.append(vector1)
                i += 1
        else:
            merged_vectors.append(vector1)
            i += 1

    # If there's an odd number of vectors, add the last vector
    if i == len(vectors) - 1:
        merged_vectors.append(vectors[-1])

    return merged_vectors

def visualize_results(lines, noisy_points, fitted_lines, extents, intersections, merged_vectors):
    (x_min, x_max, x_padding), (y_min, y_max, y_padding) = extents

    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)

    if intersections:
        plt.scatter(np.array(intersections)[:, 0], np.array(intersections)[:, 1], c='black', marker='o', label="Intersections")

    # Plotting the merged vectors
    for start_point, end_point in merged_vectors:
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'b', label="Merged Vector")

    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Connected Vectors from DXF File (Merged)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    lines = load_lines_from_dxf(file_path)
    noisy_points = generate_noisy_points(lines)
    fitted_lines = [fit_line_ransac(noisy_points[i * num_points_per_line:(i + 1) * num_points_per_line])[:2] for i in range(len(lines))]

    extents = calculate_extents(lines)
    intersections = compute_intersections(fitted_lines, extents)
    filtered_vectors = create_filtered_vectors_from_intersections(fitted_lines, intersections, noisy_points, min_distance=0.1)
    merged_vectors = merge_vectors(filtered_vectors)

    visualize_results(lines, noisy_points, fitted_lines, extents, intersections, merged_vectors)
