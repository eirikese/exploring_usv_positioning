import numpy as np

loaded_arrays = np.load("MultiMatrix.npz")

# print(loaded_arrays.files)
print("camMatrix:")
print(loaded_arrays["camMatrix"])
print("distCoef:")
print(loaded_arrays["distCoef"])
