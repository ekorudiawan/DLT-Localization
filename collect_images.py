from pygrabber.dshow_graph import FilterGraph
import cv2
import datetime

graph = FilterGraph()

# Print available camera devices
print("Available devices ", graph.get_input_devices())

try:
    device = graph.get_input_devices().index("Logitech Webcam C930e")
except ValueError as e:
    # use default camera if the name of the camera that I want to use is not in my list
    device = graph.get_input_devices().index("USB2.0 HD UVC WebCam")

cap = cv2.VideoCapture(device)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Camera", frame)

    # Check for the 'g' key to save the image
    if cv2.waitKey(1) & 0xFF == ord("g"):
        # Generate a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Save the image with a timestamped filename
        filename = "saved_image_{}.jpg".format(timestamp)
        cv2.imwrite(filename, frame)
        print("Image saved as", filename)

    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
