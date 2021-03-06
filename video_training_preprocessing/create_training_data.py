import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import math
import numpy as np
import os
import pickle

# Compute the angle between two joints by using three points.
def get_angle(p1, p2, p3, side):
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    if side == 'left':
        return -math.sin(radians), math.cos(radians)
    else:
        return math.sin(radians), math.cos(radians)

# Get the sines and cosines for each frame in the video and store them in a matrix.
def get_angles_video(poseEstimator, category, video_name):
    if not video_name.endswith(".avi"):
        video_name = video_name + ".avi"
    cap = cv2.VideoCapture("cropped_vids/{}/{}".format(category, video_name))
     
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    
    correct_frames = 0
    total_frames = 0
    # Matrix to store the sines and cosines.
    angles_matrix = np.zeros([1, 8])
    
    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            total_frames +=1
            # Get the skeletons.
            skeletons = poseEstimator.inference(frame, upsample_size=4.0)
            if len(skeletons) > 0:
                correct_skeleton = skeletons[0]
                
                # Only use the data is the hands, elbows, shoulders and the neck is detected.
                if all(part in correct_skeleton.body_parts for part in [1, 2, 3, 4, 5, 6, 7]):
                    
                    neck = correct_skeleton.body_parts[1]
                    rshoulder = correct_skeleton.body_parts[2]
                    relbow = correct_skeleton.body_parts[3]
                    rwrist = correct_skeleton.body_parts[4]
                    lshoulder = correct_skeleton.body_parts[5]
                    lelbow = correct_skeleton.body_parts[6]
                    lwrist = correct_skeleton.body_parts[7]
                    
                    # Get the sine and cosine of the combinations.
                    angle_lelbow_sin, angle_lelbow_cos = get_angle(lwrist, lelbow, lshoulder, "left")
                    angle_relbow_sin, angle_relbow_cos = get_angle(rwrist, relbow, rshoulder, "right")
                    angle_lshoulder_sin, angle_lshoulder_cos = get_angle(lelbow, lshoulder, neck, "left")
                    angle_rshoulder_sin, angle_rshoulder_cos = get_angle(relbow, rshoulder, neck, "right")
                    
                    # Draw the skeleton on the image.
                    image_drawn = TfPoseEstimator.draw_humans(frame, skeletons, imgcopy=False)
                    image_drawn = cv2.resize(image_drawn, (0,0), fx=0.5, fy=0.5)
                    
                    angles = np.array([angle_relbow_sin, angle_relbow_cos, angle_rshoulder_sin, angle_rshoulder_cos, angle_lshoulder_sin, angle_lshoulder_cos, angle_lelbow_sin, angle_lelbow_cos])
                    # Replace matrix if first correct frame.
                    if correct_frames == 0:
                        angles_matrix[0] = angles
                    else:
                        # Stack if not first correct frame.
                        angles_matrix = np.vstack([angles_matrix, angles])
                    correct_frames += 1
                    
                    # Display the resulting frame.
                    cv2.imshow('Frame', image_drawn)
                    
                    # Exit.
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
     
        # Finished.
        else:
            return angles_matrix

    cap.release()

def get_training_data(path_video_angles, full_angles_path, video_names):
    if not os.path.exists(path_video_angles):
        os.makedirs(path_video_angles)

    if os.path.isfile(full_angles_path) :
        with open(full_angles_path, 'rb') as f:
            angles_per_video = pickle.load(f)
    else:
        angles_per_video = dict()
        
    # Show each frame in the same window, as if it is a video.
    cv2.namedWindow('Frame')
    cv2.startWindowThread()
    
    # Create a pose estimator.
    poseEstimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(400, 400))

    num_videos = len(video_names)
    i = 0
    for video_name in video_names:
        if video_name.endswith(".avi"):
            i += 1
            print("Processing video {}, {}/{}".format(video_name, i, num_videos))
            if video_name in angles_per_video:
                print("Video already processed")
                continue
            angles_matrix = get_angles_video(poseEstimator, category, video_name)
            angles_per_video[video_name] = angles_matrix

    with open(full_angles_path, 'wb') as f:
        pickle.dump(angles_per_video, f)
        
    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path_video_angles = "../training_angles"
    category = "noise"
    file_video_angles = "{}.pkl".format(category)
    full_angles_path = "{}/{}".format(path_video_angles, file_video_angles)

    video_names = [f for f in os.listdir("cropped_vids/{}".format(category)) if f.endswith(".avi")]
     
    get_training_data(path_video_angles, full_angles_path, video_names)
        

