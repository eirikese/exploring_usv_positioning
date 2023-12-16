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
    return [entity for entity in msp if entity.dxftype() == 'LINE']

def generate_noisy_points(lines, num_points=num_points_per_line, noise_std=noise_std):
    points = []
    for line in lines:
        start, end = np.array(line.dxf.start)[:2], np.array(line.dxf.end)[:2]
        linear_points = np.linspace(start, end, num_points)
        noise = np.random.normal(scale=noise_std, size=(num_points, 2))
        points.extend(linear_points + noise)
    return np.array(points)

def fit_line_ransac(points):
    X, y = points[:, 0].reshape(-1, 1), points[:, 1]
    ransac = RANSACRegressor().fit(X, y)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

def find_intersection(m1, b1, m2, b2):
    if m1 == m2:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def visualize_results(lines, noisy_points, fitted_lines):
    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)
    
    for line in lines:
        plt.plot([line.dxf.start[0], line.dxf.end[0]], [line.dxf.start[1], line.dxf.end[1]], 'g-', label="Original Line")

    intersections = []
    for i, (m1, b1) in enumerate(fitted_lines):
        for j, (m2, b2) in enumerate(fitted_lines):
            if i >= j:  # Avoid duplicate checks and self-intersection
                continue
            intersection = find_intersection(m1, b1, m2, b2)
            if intersection:
                intersections.append(intersection)
        
        x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
        plt.plot(x, m1 * x + b1, 'r--', label=f"Fitted Line {i+1}")
    
    if intersections:
        intersections = np.array(intersections)
        plt.scatter(intersections[:, 0], intersections[:, 1], c='blue', marker='x', s=100, label="Intersections")
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Connected Vectors from DXF File')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    lines = load_lines_from_dxf(file_path)
    noisy_points = generate_noisy_points(lines)
    
    fitted_lines = [fit_line_ransac(noisy_points[i*num_points_per_line: (i+1)*num_points_per_line]) for i in range(len(lines))]
    
    visualize_results(lines, noisy_points, fitted_lines)
