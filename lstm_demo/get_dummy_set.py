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

dataset_x, dataset_y = get_small_dataset_data()  
    
#    #print(len(new_dataset), 'items')
#    #for i in range(len(new_dataset_x)): 
#    #    print("{}: {}".format(new_dataset[i][0], new_dataset[i][1]))
#
#    return new_dataset, DATASET_Y


#from collections import defaultdict
#
## Returns a dict id = [[[points1],[points2],...],...]
#def get_small_dataset_sequences():
#    
#    # The ratios of movement in a single sequence
#    OFFSET_RATIOS = [0.98, 1.01, 1.05, 1.1, 1.15, 1.12, 1.1, 1.08, 1.09]
#        
#    dataset, classes = get_small_dataset_data()
#    
#    new_dataset = defaultdict(lambda: [])
#    for x, y in dataset:
#        
#        new_sequence = []
#        for offset_ratio in OFFSET_RATIOS:
#            new_sequence.append([round(el*offset_ratio,2) for el in x])
#            
#        new_dataset[y].append(new_sequence)
#    
#    return new_dataset, classes, len(OFFSET_RATIOS)
#