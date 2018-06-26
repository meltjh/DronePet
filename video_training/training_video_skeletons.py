import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
import math
import numpy as np
import os
import pickle

def get_angle(p1, p2, p3, side):
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    if side == 'left':
        return -math.sin(radians)
    else:
        return math.sin(radians)

def get_angles(skeleton):
    if all(part in skeleton.body_parts for part in [1, 2, 3, 4, 5, 6, 7]):
                    
        neck = skeleton.body_parts[1]
        rshoulder = skeleton.body_parts[2]
        relbow = skeleton.body_parts[3]
        rwrist = skeleton.body_parts[4]
        lshoulder = skeleton.body_parts[5]
        lelbow = skeleton.body_parts[6]
        lwrist = skeleton.body_parts[7]
        
        angle_lelbow = get_angle(lwrist, lelbow, lshoulder, "left")
        angle_relbow = get_angle(rwrist, relbow, rshoulder, "right")
        angle_lshoulder = get_angle(lelbow, lshoulder, neck, "left")
        angle_rshoulder = get_angle(relbow, rshoulder, neck, "right")
    
        return [angle_relbow, angle_rshoulder, angle_lshoulder, angle_lelbow]
    return None

def get_angles_video(poseEstimator, category, video_name):
    if not video_name.endswith(".avi"):
        video_name = video_name + ".avi"
    cap = cv2.VideoCapture("cropped_vids/{}/{}".format(category, video_name))
     
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    
    correct_frames = 0
    frames = 0
    angles_matrix = np.zeros([1, 4])
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frames +=1
    #       frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
#            frame = cv2.flip(frame, 1 )
            skeletons = poseEstimator.inference(frame, upsample_size=4.0)
            if len(skeletons) > 0:
                angles_data = get_angles(skeletons[0])
                if angles_data is not None:
                    
                    image_drawn = TfPoseEstimator.draw_humans(frame, skeletons, imgcopy=False)
                    image_drawn = cv2.resize(image_drawn, (0,0), fx=0.5, fy=0.5)
                    
                    angles = np.array(angles_data)
                    if correct_frames == 0:
                        angles_matrix[0] = angles
                    else:
                        angles_matrix = np.vstack([angles_matrix, angles])
                    correct_frames += 1
                    
                    # Display the resulting frame
                    cv2.imshow('Frame', image_drawn)
                    
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
     
        # Break the loop
        else:
            print(correct_frames)
            return angles_matrix
    
    # When everything done, release the video capture object
    cap.release()

if __name__ == "__main__":
    path_video_angles = "video_angles"
    category = "down"
    file_video_angles = "../{}.pkl".format(category)
    full_angles_path = "{}/{}".format(path_video_angles, file_video_angles)
    
    if not os.path.exists(path_video_angles):
        os.makedirs(path_video_angles)

    if os.path.isfile(full_angles_path) :
        with open(full_angles_path, 'rb') as f:
            angles_per_video = pickle.load(f)
    else:
        angles_per_video = dict()
    
    cv2.namedWindow('Frame')
    cv2.startWindowThread()
    
    poseEstimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(400, 400))
    
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
#    video_names = ["short_S001C001P005R002A023_rgb", "short_S001C003P004R001A023_rgb"]
    video_names = [f for f in os.listdir("cropped_vids/{}".format(category)) if f.endswith(".avi")]
    print(video_names)
#    video_names = ["short_S001C001P001R001A010_rgb"]
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
        

