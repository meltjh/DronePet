# Preprocessing of the training videos. Displays a single frame, so that it 
# can be decided when the movie should start/finish. This works in Spyder, 
# while showing whole movies does not. The frame is cropped, because only the
# person performing the action is important for training and the shorter 
# movie is saved. 

import cv2
import matplotlib.pyplot as plt
import os

def show_begin_end_frame(begin_frame, max_frame, original_vids_path, cropped_video_path, cap_dim1, cap_dim2, show_first=True):
    
    cap = cv2.VideoCapture(original_vids_path)
    fourcc = cv2.VideoWriter_fourcc(*'jpeg')
    out = cv2.VideoWriter(cropped_video_path, fourcc, 30.0, (cap_dim1, cap_dim2), True)
    
    # Check if video opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    total_frames = 0
    num_frames_after = 0
    
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if total_frames >= begin_frame and total_frames <= max_frame:
                # Crop to get the center of the frame.
                frame = frame[0:720, 100:900]
                out.write(frame)
    
                # Resize for displaying.
                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                num_frames_after +=1
    
                # Display the resulting frame
                    # Show first frame of cropped video.
                if show_first:
                    if total_frames == begin_frame:
                        print("First frame")
                        plt.imshow(frame)
                        plt.show

                # Show last frame of cropped video.
                else:
                    if total_frames == max_frame - 1:
                        print("Last frame")
                        plt.imshow(frame)
                        plt.show
            
            total_frames += 1

        # Break the loop
        else:
            print("Number of frames before: ", total_frames)
            print("Number of frames after: ", num_frames_after)
            print("Begin frame: ", begin_frame)
            if max_frame >= total_frames:
                last_frame = total_frames - 1
            else:
                last_frame = max_frame
            print("End frame: ", last_frame)
            break
 
    cap.release()
    out.release()

if __name__ == "__main__":

    video_name = "noise_5"
    category = "noise"
    original_vids_path = "full_vids/{}/{}.mov".format(category, video_name)

    # Create directory for cropped videos.
    cropped_category_directory = "cropped_vids/{}".format(category)
    if not os.path.exists(cropped_category_directory):
        os.makedirs(cropped_category_directory)
    
    num_sub_vid = 1
    cropped_video_path = "{}/short_{}_{}.avi".format(cropped_category_directory, video_name, num_sub_vid)
    begin_frame = 0
    end_frame = 100
    
    cap_dim1 = 800
    cap_dim2 = 720
    
    show_begin_end_frame(begin_frame, end_frame, original_vids_path, cropped_video_path, cap_dim1, cap_dim2, False)
