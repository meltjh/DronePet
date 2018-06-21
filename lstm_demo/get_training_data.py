import os
import pickle
import get_dummy_set as gds
import numpy as np

def get_training_set(directory, sub_vid_length):
    file_names = os.listdir(directory)
    num_classes = len(file_names)
    
    dataset_y = None
    dataset_x = None

    for file_name in file_names:
        with open("{}/{}".format(directory, file_name), 'rb') as f:
            angles_per_video = pickle.load(f)
        
        if file_name == "23_waving.pkl":
            class_num = 0
        elif file_name == "23_waving_flipped.pkl":
            class_num = 1
        else:
            class_num = 2

        one_hot_y = np.zeros(num_classes)
        one_hot_y[class_num] = 1
        
        for key, matrix in angles_per_video.items():
            cropped = gds.crop_video(matrix, sub_vid_length)
            num_sub_vids = len(cropped)
            
            
            x_block = np.stack(cropped)
            one_hot_y_block = np.tile(one_hot_y, (num_sub_vids, 1)) # (46, 2)
            if dataset_x is None:
                dataset_x = x_block
                dataset_y = one_hot_y_block
            else:
                dataset_x = np.vstack([dataset_x, x_block])
                dataset_y = np.vstack([dataset_y, one_hot_y_block])

    return dataset_x, dataset_y

#get_training_set("../video_angles")