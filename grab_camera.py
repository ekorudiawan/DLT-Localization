import cv2 as cv
import numpy as np
import cv2.aruco as aruco
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()

print("Available camera devices ", graph.get_input_devices())

try:
    device = graph.get_input_devices().index("Logitech Webcam C930e")
except ValueError as e:
    device = graph.get_input_devices().index("USB2.0 HD UVC WebCam")

# Load the saved camera parameters
npzfile = np.load("camera_parameters.npz")
mtx = npzfile["mtx"]
dist = npzfile["dist"]

# Initialize ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

# Initialize ArUco parameters
parameters = aruco.DetectorParameters_create()

print("Device ", device)

# Initialize OpenCV's video capture
cap = cv.VideoCapture(0)

while True:
    # Initialize marker detection flags
    marker_detected = [False] * 6

    # Initialize list to store center point locations
    center_points = [None] * 6

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Undistort the frame using the camera parameters
    undistorted_frame = cv.undistort(frame, mtx, dist)

    # Detect ArUco markers
    gray = cv.cvtColor(undistorted_frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters
    )

    # Draw detected markers and check for target marker detection
    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] >= 0 and ids[i][0] <= 5:
                marker_detected[ids[i][0]] = True
                # Get center point coordinates and print them
                center = np.mean(corners[i][0], axis=0)
                center_x, center_y = map(int, center)
                center_points[ids[i][0]] = (center_x, center_y)
                # print(f"ArUco ID: {ids[i][0]}, Center: ({center_x}, {center_y})")

    # Draw markers if all target markers are detected
    if all(marker_detected):
        # print("Center points ", center_points)
        aruco.drawDetectedMarkers(undistorted_frame, corners)
        # Estimate camera pose
        # x => left - right
        # y => up - down
        # z => pointing forward
        object_points = np.array(
            [
                [-47.5, -87.5, 800],
                [47.5, -87.5, 800],
                [-47.5, 0, 800],
                [47.5, 0, 800],
                [-47.5, 87.5, 800],
                [47.5, 87.5, 800],
            ],
            dtype=np.float32,
        )
        image_points = np.array([center_points[i] for i in range(6)], dtype=np.float32)
        _, rvecs, tvecs = cv.solvePnP(object_points, image_points, mtx, dist)

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv.Rodrigues(rvecs)

        # Inverse rotation matrix and translation vector to get camera pose in world coordinate
        camera_pose_world = -np.dot(rotation_matrix.T, tvecs)

        # Print camera pose in world coordinate
        # print("Camera Pose (World Coordinate):")
        # print(camera_pose_world.flatten())

        xx = camera_pose_world.flatten()[2]
        yy = camera_pose_world.flatten()[0]
        zz = camera_pose_world.flatten()[1]
        print("x =", xx, "y = ", yy, "z =", zz)

    # Display the undistorted frame with detected markers
    cv.imshow("Undistorted", undistorted_frame)

    # Check for the 'q' key to quit the program
    if cv.waitKey(100) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()
cv.destroyAllWindows()
