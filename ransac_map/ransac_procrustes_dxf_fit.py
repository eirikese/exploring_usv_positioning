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
    inliers = np.array(ransac.inlier_mask_)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_, inliers


def calculate_centroid(lines):
    sum_x, sum_y = 0, 0
    num_points = 0

    for line in lines:
        sum_x += (line.dxf.start[0] + line.dxf.end[0]) / 2
        sum_y += (line.dxf.start[1] + line.dxf.end[1]) / 2
        num_points += 2

    return sum_x / num_points, sum_y / num_points

def translate_lines(lines, dx, dy):
    translated_lines = []

    for line in lines:
        start = line.dxf.start
        end = line.dxf.end
        translated_start = (start[0] + dx, start[1] + dy)
        translated_end = (end[0] + dx, end[1] + dy)
        translated_lines.append((translated_start, translated_end))

    return translated_lines

def visualize_results(lines, noisy_points, fitted_lines=None, translated_dxf=None):
    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], label="Noisy Points", alpha=0.6)

    for line in lines:
        plt.plot([line.dxf.start[0], line.dxf.end[0]], [line.dxf.start[1], line.dxf.end[1]], 'g-', label="Original DXF")

    if fitted_lines:
        for m, b in fitted_lines:
            x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)
            y = m * x + b
            plt.plot(x, y, 'r--', label="Fitted Line")

    if translated_dxf:
        for translated_line in translated_dxf:
            plt.plot([translated_line[0][0], translated_line[1][0]], [translated_line[0][1], translated_line[1][1]], 'o:', label="Translated DXF")


    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Connected Vectors from DXF File')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    lines = load_lines_from_dxf(file_path)
    noisy_points = generate_noisy_points(lines)
    fitted_lines = []
    for i in range(len(lines)):
        subset_points = noisy_points[i * num_points_per_line:(i + 1) * num_points_per_line]
        m, b, _ = fit_line_ransac(subset_points)
        fitted_lines.append((m, b))

    centroid_dxf = calculate_centroid(lines)
    centroid_ransac = (np.mean([line[0] for line in fitted_lines]), np.mean([line[1] for line in fitted_lines]))

    dx = centroid_ransac[0] - centroid_dxf[0]
    dy = centroid_ransac[1] - centroid_dxf[1]

    translated_dxf = translate_lines(lines, dx, dy)

    visualize_results(lines, noisy_points, fitted_lines, translated_dxf)
