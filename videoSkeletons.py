import numpy as np
import cv2
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('S001C001P003R001A023_rgb.avi')
poseEstimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(540, 960))

 
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
 
cv2.namedWindow('Frame')
cv2.startWindowThread()

i = 0

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    i += 1
    if i % 3 == 0:
        if ret:
            frame = frame[140:940, 560:1360] #height, width
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            skeletons = poseEstimator.inference(frame, upsample_size=4.0)
            image_drawn = TfPoseEstimator.draw_humans(frame, skeletons, imgcopy=False)

            # Display the resulting frame
            cv2.imshow('Frame', image_drawn)
     
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
 
        # Break the loop
        else:
            break
 
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
