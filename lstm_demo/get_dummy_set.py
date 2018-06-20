import numpy as np


OFFSET_RATIOS = np.arange(0.9, 1.1, 0.01) # 21

OFFSETS = np.arange(-0.2, 0.2, 0.01) # 40

X_PERFECTS = [[1, 0, 0, -1],
             [1, 0, 0, 1],
             [-1, 0, 0, 1],
             [-1, 0, 0, -1],
             [0, 0, 0, 0],
             [0, -1, 1, 0],
             [0, 1, -1, 0],
             [0, -1, -1, 0],
             [0, 1, 1, 0]]

Y_CLASSES = [0, # Left hand up, Right hand down
             1, # Both hands up
             2, # Left hand down, Right hand up
             3, # Both hands down
             4, # Both hands straight
             5, # Left arm down, right arm up
             6, # Left arm up, right arm down
             7, # Both arms up
             8  # Both arms down
             ]

def get_video_frames(frame_initial):
    frames = []
    for offset_ratio in OFFSET_RATIOS:
        frame_offset = [joint*offset_ratio for joint in frame_initial]
        frames.append(frame_offset)
    
    return frames
    
# Returns offsets*4+1 frames
def get_initial_frames(frame):
    initial_frames = []
    initial_frames.append(frame) # Eerste de 0
    for offset in OFFSETS:
        for i_joint in range(len(frame)):
            new_frame = list(frame)
            new_frame[i_joint] = new_frame[i_joint] + offset
            initial_frames.append(new_frame)
    
    return initial_frames

def get_small_dataset_data():
    num_joints = 4
    num_classes = len(Y_CLASSES)
    num_videos = (len(OFFSETS)*num_joints+1)*num_classes
    num_frames = len(OFFSET_RATIOS)
    
#    print(num_joints, num_classes, num_videos, num_frames)
    
    i = 0
    dataset_x = np.zeros([num_videos, num_frames, num_joints])
    dataset_y = np.zeros([num_videos, num_classes])
    for i_class in range(len(Y_CLASSES)):
        perfect_frame = X_PERFECTS[i_class]
        initial_frames = get_initial_frames(perfect_frame)
        for initial_frame in initial_frames:
            
            video_frames = get_video_frames(initial_frame)
            
            dataset_x[i] = video_frames
            dataset_y[i,i_class] = 1
            
            i += 1
 
    return dataset_x, dataset_y

#dataset_x, dataset_y = get_small_dataset_data()  
    
def crop_video(frames, length):
    if len(frames) < length:
        print("Less than {} frames available ({}).", length, len(frames))
    
    cropped_videos = []
    for i in range(len(frames) - length + 1):
        cropped_videos.append(frames[i:i+length])
    return cropped_videos

def crop_videos(dataset_x, dataset_y, length):
    _, num_frames, num_joints = dataset_x.shape
    num_videos, num_classes = dataset_y.shape

    num_cropped_videos_per_video= num_frames - length + 1
    
    dataset_x_new = np.zeros([num_videos*num_cropped_videos_per_video, length, num_joints])
    dataset_y_new = np.zeros([num_videos*num_cropped_videos_per_video, num_classes])
    
    for i in range(num_videos):
        frames = dataset_x[i]
        cropped_videos = crop_video(frames, length)  
        dataset_x_new[i*num_cropped_videos_per_video:(i+1)*num_cropped_videos_per_video] = cropped_videos
        
        # Create a matrix of one-hot-vector [0,...,1,...0] for num_cropped_videos_per_video times.
        dataset_y_block = np.tile(dataset_y[i], (num_cropped_videos_per_video,1))
        dataset_y_new[i*num_cropped_videos_per_video:(i+1)*num_cropped_videos_per_video] = dataset_y_block
        
    return dataset_x_new, dataset_y_new
    
    
#frames = [[1,2],[3,4],[5,6],[7,8],[9,10],[10,11]]
#length = 4
##print(crop_video(frames, length))
#
#
#
#dataset_x = np.zeros([2, 6, 2])
#dataset_y = np.zeros([2, 7])
#
#
#dataset_x[0] = frames
#dataset_x[1] = frames
#
#dataset_y[0] = [0,0,1,0,0,0,0]
#dataset_y[1] = [0,0,0,1,0,0,0]
#
#dataset_x_new, dataset_y_new = crop_videos(dataset_x, dataset_y, length)
#print(dataset_x_new, dataset_y_new)