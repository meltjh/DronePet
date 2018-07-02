import pickle

def crop_matrix_to_fixed_size(directory, file_name, fixed_size):
    new_matrix_dict = dict()
    with open("{}/{}".format(directory, file_name), 'rb') as f:
        matrices = pickle.load(f)
        for video_name, matrix in matrices.items():
            print(video_name)
            print("Old", matrix.shape)
            if video_name == "short_half_circle_up_1_1.avi":
                new_matrix = matrix[2:68, :]
            if video_name == "short_half_circle_up_2_1.avi":
                new_matrix = matrix[10:76, :]
            if video_name == "short_half_circle_up_3_1.avi":
                new_matrix = matrix[14:80, :]
            if video_name == "short_half_circle_up_4_1.avi":
                new_matrix = matrix[:, :]
            if video_name == "short_half_circle_up_5_1.avi":
                new_matrix = matrix[2:68, :]
            
            print("New", new_matrix.shape)
            print("\n")
            
            new_matrix_dict[video_name] = new_matrix
        
    with open("{}/{}_cropped".format(directory, file_name), 'wb') as f:
        pickle.dump(new_matrix_dict, f)
    return new_matrix_dict
        
crop_matrix_to_fixed_size("../video_angles_new", "half_circle_up.pkl", 66)
    