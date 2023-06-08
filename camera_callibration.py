import numpy as np
import cv2 as cv
import glob

# Number of inner corners per a chessboard row and column
corner_row = 6
corner_column = 10

# The size of each square in the chessboard
# in mm
square_size = 25

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corner_row * corner_column, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_column, 0:corner_row].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob("*.jpg")

for fname in images:
    print("Filename ", fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (corner_column, corner_row), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (corner_column, corner_row), corners2, ret)
        cv.imshow("img", img)
        key = cv.waitKey(0)
        if key == ord("n"):
            continue
        else:
            break

cv.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
# Save camera parameters to a file
np.savez("camera_parameters.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

print("Camera parameters saved to 'camera_parameters.npz' file.")
