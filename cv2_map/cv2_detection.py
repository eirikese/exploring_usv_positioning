import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to load the template from a PNG file
def load_template(template_path):
    return cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

# Function to process the target image
def process_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to detect the template shape in the target image
def detect_shape(template, target):
    res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val > 0.5:  # Adjust the threshold as necessary
        top_left = max_loc
        h, w = template.shape
        return (top_left[0], top_left[1], w, h)
    else:
        return None

def main():
    template_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\template_data\image_0.png"
    target_path = r"C:\Users\eirik\OneDrive - NTNU\Semester 9\Prosjektoppgave\cv2_map\evaluation_data\image_0.png"
    
    template = load_template(template_path)
    target = process_image(target_path)
    detection = detect_shape(template, target)

    plt.subplot(121)
    plt.imshow(target, cmap='gray')
    if detection:
        print("Detected :-)")
        x, y, w, h = detection
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none'))
    plt.title('Detection')

    plt.subplot(122)
    plt.imshow(template, cmap='gray')
    plt.title('Template')

    plt.show()

if __name__ == "__main__":
    main()
