# Display a single frame, so that it can be decided when the movie should start/finish.
# This works in Spyder, while showing whole movies does not.
# The frame is cropped and the shorter movie is saved.

import cv2
import matplotlib.pyplot as plt
import os

# Create a VideoCapture object and read from input file
video_name = "down_2"
category = "down"
cap = cv2.VideoCapture("full_vids/{}/{}.mov".format(category, video_name))
fourcc = cv2.VideoWriter_fourcc(*'jpeg')

# Create directory for cropped videos.
cropped_vids_directory = "cropped_vids/{}".format(category)
if not os.path.exists(cropped_vids_directory):
    os.makedirs(cropped_vids_directory)


cap_dim1 = 800 #800
cap_dim2 = 720


sub_vid = 4
out = cv2.VideoWriter("{}/short_{}_{}.avi".format(cropped_vids_directory, video_name, sub_vid), fourcc, 30.0, 
                      (cap_dim1, cap_dim2), True)
 
# Check if video opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
 
cv2.namedWindow('Frame')
cv2.startWindowThread()

i = 0
num_frames = 0
begin_frame = 260
max_frame = 325

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        if i >= begin_frame and i <= max_frame:
            # Crop to get the center of the frame.
#            frame = frame[140:940, 560:1360]
            frame = cv2.flip(frame, 1)
            frame = frame[0:720, 180:980]
            out.write(frame)

            # Resize for displaying.
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            num_frames +=1

            # Display the resulting frame
            if i == begin_frame: # When looking at the beginning of the video.
#            if i == max_frame - 1: # When looking at end of video.
                plt.imshow(frame)
                plt.show
#                break
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        i += 1
 
    # Break the loop
    else:
        print("Number of frames before: ", i)
        print("Number of frames after: ", num_frames)
        print("Begin frame: ", begin_frame)
        if max_frame >= i:
            last_frame = i - 1
        else:
            last_frame = max_frame
        print("End frame: ", last_frame)
        break
 
# When everything done, release.
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
