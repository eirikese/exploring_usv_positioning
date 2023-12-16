import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from matplotlib.patches import Polygon

# Define variables
file_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\vector_fitting\boot.dxf"
num_points_per_line = 25
noise_std = 0.01
MAX_CLUSTERS = 10

def load_lines_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    return [entity for entity in msp if entity.dxftype() == 'LINE']

def generate_noisy_points(lines, num_points=num_points_per_line, noise_std=noise_std):
    all_points = []
    for line in lines:
        start, end = np.array(line.dxf.start)[:2], np.array(line.dxf.end)[:2]
        points_along_edge = np.linspace(start, end, num_points)
        noise = np.random.normal(scale=noise_std, size=(num_points, 2))
        noisy_points = points_along_edge + noise
        all_points.extend(noisy_points)
    return np.array(all_points)

def fit_line_ransac(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac = RANSACRegressor().fit(X, y)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

def get_clusters_with_high_aspect_ratio(points, min_samples=2, eps=0.1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    high_aspect_ratio_clusters = []
    for cluster_label in set(labels):
        if cluster_label == -1:
            continue
        cluster_points = points[labels == cluster_label]
        min_x, min_y = np.min(cluster_points, axis=0)
        max_x, max_y = np.max(cluster_points, axis=0)
        width = max_x - min_x
        height = max_y - min_y

        if height == 0 or width == 0:
            continue

        aspect_ratio = max(width/height, height/width)
        if aspect_ratio > 10:
            high_aspect_ratio_clusters.append(cluster_points)
    return high_aspect_ratio_clusters

def remove_detected_points(points, cluster_points):
    return np.array([p for p in points if not any(np.all(p == cp) for cp in cluster_points)])

def visualize_results(lines, noisy_points, fitted_lines, clusters=None):
    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)
    for line in lines:
        plt.plot([line.dxf.start[0], line.dxf.end[0]], [line.dxf.start[1], line.dxf.end[1]], 'g-', label="Original Line")
    if fitted_lines:
        for m, b in fitted_lines:
            x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
            y = m * x + b
            plt.plot(x, y, 'r--', label="Fitted Line")
    if clusters:
        for cluster in clusters:
            poly = Polygon(cluster, edgecolor='blue', alpha=0.4)
            plt.gca().add_patch(poly)
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
    clusters = []
    fitted_lines = []
    for _ in range(MAX_CLUSTERS):
        high_aspect_ratio_clusters = get_clusters_with_high_aspect_ratio(noisy_points)
        if not high_aspect_ratio_clusters:
            break
        cluster = high_aspect_ratio_clusters[0]
        clusters.append(cluster)
        m, b = fit_line_ransac(cluster)
        fitted_lines.append((m, b))
        noisy_points = remove_detected_points(noisy_points, cluster)
    visualize_results(lines, noisy_points, fitted_lines, clusters)
