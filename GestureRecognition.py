import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from Actions import Action
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
import math
import timeit




class Bodyparts(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18
   
class LSTMGestureRecognition:
    def __init__(self):
        
        self.sess = tf.Session()
        loader = tf.train.import_meta_graph("./lstm_model/model.meta")
        loader.restore(self.sess, tf.train.latest_checkpoint("./lstm_model"))
          
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.bias = graph.get_tensor_by_name("out_bias:0")
        self.temp_prediction = graph.get_tensor_by_name("temp_prediction:0")
        
        
        video_length = 5
        amount_of_joints = 4
        self.frames_queue = deque([[0]*amount_of_joints]*video_length, video_length)
        
    def predict(self):      
        feed_dict ={self.x:[self.frames_queue]}      
        prediction = self.temp_prediction + self.bias
        
        label = self.sess.run(tf.argmax(prediction, 1), feed_dict)
        
        return label
    
    def append_skeleton(self, skeleton):
        frame = self.get_angles(skeleton)
        if frame is not None:
            self.frames_queue.appendleft(frame)
         
    def get_angle(self, p1, p2, p3, side):
        radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
        if side == 'left':
            return -math.sin(radians)
        else:
            return math.sin(radians)
    
    def get_angles(self, skeleton):
        if all(part in skeleton.body_parts for part in [1, 2, 3, 4, 5, 6, 7]):
                        
            neck = skeleton.body_parts[1]
            rshoulder = skeleton.body_parts[2]
            relbow = skeleton.body_parts[3]
            rwrist = skeleton.body_parts[4]
            lshoulder = skeleton.body_parts[5]
            lelbow = skeleton.body_parts[6]
            lwrist = skeleton.body_parts[7]
            
            angle_lelbow = self.get_angle(lwrist, lelbow, lshoulder, "left")
            angle_relbow = self.get_angle(rwrist, relbow, rshoulder, "right")
            angle_lshoulder = self.get_angle(lelbow, lshoulder, neck, "left")
            angle_rshoulder = self.get_angle(relbow, rshoulder, neck, "right")
        
            return [angle_relbow, angle_rshoulder, angle_lshoulder, angle_lelbow]
        return None
        
class GestureRecognition:
    
    def __init__(self, faceRecognition, droneController):
        
    
        # Load the trained model for different ratios
        self.scales = [(11,9)]#[(12,8), (11,9), (11,10), (10,10)]
        self.pose_estimators = []
        self.pose_estimators_ratios = []
        for w, h in self.scales:
            ratio = w/h
            self.pose_estimators_ratios.append(ratio)
            self.pose_estimators.append(TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w*16, h*16)))
        self.pose_estimators_ratios = np.asarray(self.pose_estimators_ratios)
    
        # Face recognition is used in id_skeleton
        self.faceRecognition = faceRecognition
        self.droneController = droneController
        
        self.lstm_gesture_recognition = LSTMGestureRecognition()
            
            
    def relative_to_absolute_pos(self, patch_shape, patch_margins, pos):
        x = pos.x
        y = pos.y
        
        [margin_top, margin_bottom, margin_left, margin_right] =  patch_margins   
        # If there was some margin, calculate the relative positions and update the skeleton.
        if max(margin_top, margin_bottom, margin_left, margin_right) > 0:
            
            [patch_h, patch_w, _] = patch_shape
            offset_x = margin_left
            offset_y = margin_top
            full_w = margin_left + patch_w + margin_right
            full_h = margin_top + patch_h + margin_bottom
        
            return (x*patch_w+offset_x)/full_w, (y*patch_h+offset_y)/full_h
            
        return x, y
        
    def determine_camera_movements(self, skeleton, patch_shape, patch_margins):
        nose_x_scale, nose_y_scale = self.relative_to_absolute_pos(patch_shape, patch_margins, skeleton.body_parts[0])
        
        
#        print("NOSE SCALE X", nose_x_scale)
        middle = 0.5
        margin = 0.05
        
        command = Action.NOTHING
        if nose_x_scale > middle + margin:
#            offset = nose_x_scale - (middle + margin)
            offset = nose_x_scale - middle
            command = Action.LOOK_RIGHT
#            print("GO TO RIGHT")
        elif nose_x_scale < middle - margin:
#            offset = (middle - margin) - nose_x_scale
            offset = middle  - nose_x_scale
            command = Action.LOOK_LEFT
#            print("GO TO LEFT")
        else:
            offset = 0
        
        if offset != 0:
#            print("OFFSET X" , offset)
            value = (((offset+1)**11)-1)
            self.droneController.perform_action(command, value)
    

        
        
#        print("NOSE SCALE Y", nose_y_scale)
        middle = 0.4 # <.5 so see more body
        margin = 0.075
        
        command = Action.NOTHING
        if nose_y_scale > middle + margin:
#            offset = nose_y_scale - (middle + margin)
            offset = nose_y_scale - middle
            command = Action.LOOK_DOWN
#            print("GO TO DOWN")
        elif nose_y_scale < middle - margin:
#            offset = (middle - margin) - nose_y_scale
            offset = middle - nose_y_scale
            command = Action.LOOK_UP
#            print("GO TO UP")
        else:
            offset = 0
        
        if offset != 0:
#            print("OFFSET Y" , offset)
            value = (((offset+1)**9)-1)
            self.droneController.perform_action(command, value)
            
    
    

    def get_correct_pose_estimator(self, w, h):
        ratio = w/h
        idx = (np.abs(self.pose_estimators_ratios - ratio)).argmin()
        
        
        print("Ratio {} -> {} ({})".format(ratio, self.pose_estimators_ratios[idx], idx))
        
        return self.pose_estimators[idx]
    
    def get_corresponding_skeleton(self, skeletons, face_position, img_h, img_w):
        (top, bottom, left, right) = face_position
        
        for skeleton in skeletons:
            if 0 in skeleton.body_parts:
                nose = skeleton.body_parts[0]
                if left < nose.x*img_w and right > nose.x*img_w and top < nose.y*img_h and bottom > nose.y*img_h:
                    return skeleton
        
        return None
    
    # Draws the skeleton and returns the exact relative face position of the given id, or if None, the first face.
    def main(self, image_original, image_drawn, face_position, specific_face_id = None, patch_margins = (0,0,0,0)):
        if image_original is None:
            print('No image given for gesture_recognition')
            return False
        
        # Obtain the skeletons
        
        image_original_bgr = cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR)
        
        
        
        img_h, img_w, _ = image_original_bgr.shape
        tf_pose_est = self.get_correct_pose_estimator(img_w, img_h)
        
        # Downscale the image beforehand will increase the execution time, slightly.
        image_original_bgr = cv2.resize(image_original_bgr, (self.scales[0][0]*16, self.scales[0][1]*16))
        
        skeletons = tf_pose_est.inference(image_original_bgr, upsample_size=4.0)


        margin_top, margin_bottom, margin_left, margin_right = patch_margins
        face_top, face_bottom, face_left, face_right = face_position
        face_position = (face_top-margin_top, face_bottom-margin_top, face_left-margin_left, face_right-margin_left)
        correct_skeleton = self.get_corresponding_skeleton(skeletons, face_position, img_h, img_w)

        if correct_skeleton is None:
            print("No skeleton in the face position found")
            return False

        self.determine_camera_movements(correct_skeleton, image_original.shape, patch_margins)
        
        image_drawn = tf_pose_est.draw_humans(image_drawn, [correct_skeleton], imgcopy=False)
        
        
        self.lstm_gesture_recognition.append_skeleton(correct_skeleton)
        label = self.lstm_gesture_recognition.predict()
        
        if label == 0:
            print("Waving left hand")
        elif label == 1:
            print("Waving right hand")
        elif label == 2:
            print("Clapping")
        else:
            print("Other action")

        return True