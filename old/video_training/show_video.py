# Shows the video, does not work in Spyder, so it is ran in the console.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

# Create a VideoCapture object and read from input file
video_name = "S001C002P004R002A010_rgb"
category = "10_clapping"
cap = cv2.VideoCapture("full_vids/{}/{}.avi".format(category, video_name))
 
# Check if camera opened successfully.
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Window to display on.
#cv2.namedWindow('Frame')
#cv2.startWindowThread()

i = 0

# Last frame
end_frame = 80

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        i += 1
        # Resize for displaying.
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # Display the resulting frame
#        cv2.imshow('Frame', frame)
        plt.imshow(frame)
        plt.show
        break

        # Stop at last frame.
        if i == end_frame:
            break
     
        # Press Q on keyboard to  exit
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            break
 
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
# Closes all the frames
#cv2.destroyAllWindows()
