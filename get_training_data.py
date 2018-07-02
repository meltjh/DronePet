# Returns the training data of the sines and cosines together with the labels.

import os
import pickle
import numpy as np

def get_training_set(directory, skip_frame):
    file_names = os.listdir(directory)
    num_classes = len([file for file in file_names if file.endswith(".pkl")])
    
    dataset_y = None
    dataset_x = None

    for file_name in file_names:
        if file_name.endswith(".pkl"):
            with open("{}/{}".format(directory, file_name), "rb") as f:
                angles_per_video = pickle.load(f)
        
            # Get the class label.
            if file_name == "up.pkl":
                class_num = 0
            elif file_name == "down.pkl":
                class_num = 1
            elif file_name == "half_circle_up.pkl":
                class_num = 2
            elif file_name == "half_circle_down.pkl":
                class_num = 3
            elif file_name == "noise.pkl":
                class_num = 4
            
            # Create one hot vector.
            one_hot_y = np.zeros(num_classes)
            one_hot_y[class_num] = 1

            for key, matrix in angles_per_video.items():
                # Skip few frames.
                cropped = crop_skip_video(matrix, skip_frame)

                x_block = np.stack([cropped[0]])
                one_hot_y_block = np.tile(one_hot_y, (1, 1))

                if dataset_x is None:
                    dataset_x = x_block
                    dataset_y = one_hot_y_block
                else:
                    # Add to the matrix.
                    dataset_x = np.vstack([dataset_x, x_block])
                    dataset_y = np.vstack([dataset_y, one_hot_y_block])

    return dataset_x, dataset_y

# Returns 0:2:n, 1:2:n etc, to get multiple sub videos out of one.
def crop_skip_video(frames, skip_size):
    if len(frames)%skip_size != 0:
        print("Frames will not be equally long: {}%{}={}, should be 0.".format(len(frames), skip_size, len(frames)%skip_size))
    cropped_videos = []
    for i in range(skip_size):
        cropped_videos.append(frames[i:len(frames):skip_size])

    return cropped_videos

if __name__ == "__main__":
    get_training_set("../training_angles", 11)