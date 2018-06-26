import os
import pickle
import get_dummy_set as gds
import numpy as np

def get_training_set(directory, max_sub_vid_length):
    file_names = os.listdir(directory)
    num_classes = len(file_names)
    
    dataset_y = None
    dataset_x = None
    sequence_lengths = []

    for file_name in file_names:
        if file_name.endswith(".pkl"):
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
                num_frames = len(matrix)
                length_diff = max_sub_vid_length - num_frames
                if length_diff <= 0: # If smaller or equal to max time steps.
                    cropped = gds.crop_video(matrix, max_sub_vid_length)
                    num_sub_vids = len(cropped)
                    
                    x_block = np.stack(cropped)
                    hot_y_block = np.tile(one_hot_y, (num_sub_vids, 1)) # (46, 2)
                    seq_len_block =  np.tile(num_frames, (num_sub_vids))
                    
                else: # If video shorter than number of time steps.
                    x_block = np.zeros((1, max_sub_vid_length, 4))
                    temp_matrix = matrix[None, :]
                    x_block[:, :num_frames, :] = temp_matrix
                    hot_y_block = np.tile(one_hot_y, (1, 1)) # (46, 2)
                    seq_len_block = np.array([num_frames])

                if dataset_x is None:
                    dataset_x = x_block
                    dataset_y = hot_y_block
                    sequence_lengths.append(seq_len_block)
                else:
                    dataset_x = np.vstack([dataset_x, x_block])
                    dataset_y = np.vstack([dataset_y, hot_y_block])
                    sequence_lengths.append(seq_len_block)

    return dataset_x, dataset_y, sequence_lengths

#x, y, seq = get_training_set("../video_angles", 50)