import cv2
import os

# Dimensions of the chessboard
CHESSBOARD_DIM = (9, 7)

# Image counter
n = 0 

# Path for storing images
store_images_path = "calibration_images" 

# Check if the directory exists
# If not. The directory is made
if os.path.isdir(store_images_path) == False: 
    os.makedirs(store_images_path)
    print(f'"{store_images_path}" Directory is created')
else:
    print(f'"{store_images_path}" Directory already exists.')

# Define criteria for corner detection in the chessboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Function to detect chessboard corners and draw them on the image
def detect_checker_board(image, grayImage, criteria, dimension):
    ret, corners = cv2.findChessboardCorners(grayImage, dimension)
    if ret == True:
        corners1 = cv2.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv2.drawChessboardCorners(image, dimension, corners1, ret)
    return image, ret

# Starting video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# The main loop for video capture and processing
while True:
    _, frame = cap.read()
    copyFrame = frame.copy()
    gray = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)

    image, board_detected = detect_checker_board(
        frame, gray, criteria, CHESSBOARD_DIM
    )
    
    # Display the image counter
    cv2.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv2.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Show the processed and original frames
    cv2.imshow("frame", frame)
    cv2.imshow("copyFrame", copyFrame)

    # Handle inputs
    # Press s to save images for calibraton, q to quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s") and board_detected == True:
        cv2.imwrite(f"{store_images_path}/image{n}.png", copyFrame)
        print(f"saved image number {n}")
        n += 1
cap.release()
cv2.destroyAllWindows()

print("Total saved Images:", n)