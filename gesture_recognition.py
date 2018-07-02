import cv2
from enum import Enum
import math
from collections import deque
import numpy as np

import tensorflow as tf
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

from actions import Action

import time
   
class LSTMGestureRecognition:
    
    def __init__(self):    
        # Initialize the lstm and load this in the tf graph.
        self.sess = tf.Session()
        loader = tf.train.import_meta_graph("./lstm_model/model.meta")
        loader.restore(self.sess, tf.train.latest_checkpoint("./lstm_model"))
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.bias = graph.get_tensor_by_name("out_bias:0")
        self.temp_prediction = graph.get_tensor_by_name("temp_prediction:0")
        
        # The amount of frames of a video that is input for the lstm.
        video_length = 6
        # The amount of features are now 8, sine and cosine for each of the four joints.
        amount_of_joints = 8
        # The queue of frames always contain 6 frames with 8 values each.
        self.frames_queue = deque([[0]*amount_of_joints]*video_length, video_length)

    # Return the label given the current frames_queue.      
    def predict(self):      
        feed_dict ={self.x:[self.frames_queue]}      
        prediction = self.temp_prediction + self.bias
        
        label = self.sess.run(tf.argmax(prediction, 1), feed_dict)
        
        return label
    
    # The skeleton will be added to the frames queue.
    def append_skeleton(self, skeleton):
        frame = self.get_angles(skeleton)
        if frame is not None:
            self.frames_queue.append(frame)
         
    # Convert the join of p2 given p1 and p3. The side will be for left or right.
    # Note that this does not matter when training in the same way, but distinction makes the
    # interpertation easier.
    def get_angle(self, p1, p2, p3, side):
        radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
        if side == 'left':
            return -math.sin(radians), math.cos(radians)
        else:
            return math.sin(radians), math.cos(radians)
    
    # Get all 8 features for the given skeleton if all necessary noints are available.
    def get_angles(self, skeleton):
        if all(part in skeleton.body_parts for part in [1, 2, 3, 4, 5, 6, 7]):
            # Obtain all the joints
            neck = skeleton.body_parts[1]
            rshoulder = skeleton.body_parts[2]
            relbow = skeleton.body_parts[3]
            rwrist = skeleton.body_parts[4]
            lshoulder = skeleton.body_parts[5]
            lelbow = skeleton.body_parts[6]
            lwrist = skeleton.body_parts[7]
    
            # Obtain sine and cosine of the important combinations of joints.
            angle_lelbow_sin, angle_lelbow_cos = self.get_angle(lwrist, lelbow, lshoulder, "left")
            angle_relbow_sin, angle_relbow_cos = self.get_angle(rwrist, relbow, rshoulder, "right")
            angle_lshoulder_sin, angle_lshoulder_cos = self.get_angle(lelbow, lshoulder, neck, "left")
            angle_rshoulder_sin, angle_rshoulder_cos = self.get_angle(relbow, rshoulder, neck, "right")
            
            # Return all the important angles.
            return [angle_relbow_sin, angle_relbow_cos, angle_rshoulder_sin, angle_rshoulder_cos, angle_lshoulder_sin, angle_lshoulder_cos, angle_lelbow_sin, angle_lelbow_cos]
        return None   

        
class GestureRecognition:
    
    def __init__(self, faceRecognition, droneController):
        # Load the trained model for different ratios
        self.scales = [(9, 8), (10, 8), (9,9)]#[(14,8), (11,8)]
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

        # Initialize the queue
        self.first_last = 4
        self.second_last = 4
        self.third_last = 4
        
    # Returns the estimator which is closest to the given aspect ratio.
    def get_correct_pose_estimator(self, w, h):
        ratio = w/h
        idx = (np.abs(self.pose_estimators_ratios - ratio)).argmin()
        return self.pose_estimators[idx]
    
    # Return the first skeleton which has its nose within the face frame.
    def get_corresponding_skeleton(self, skeletons, face_position, img_h, img_w):
        (top, bottom, left, right) = face_position
        
        for skeleton in skeletons:
            if 0 in skeleton.body_parts:
                nose = skeleton.body_parts[0]
                if left < nose.x*img_w and right > nose.x*img_w and top < nose.y*img_h and bottom > nose.y*img_h:
                    return skeleton
        
        return None

    # Draws the skeleton and returns the exact relative face position of the given id, or if None, the first face.
    # Returns True if everything worked well. If something could not be recognized, False will be returned.
    def main(self, image_original, image_drawn, face_position, specific_face_id = None, patch_margins = (0,0,0,0)):
        if image_original is None:
            print('No image given for gesture_recognition')
            return False

        image_original_bgr = cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR)
        img_h, img_w, _ = image_original_bgr.shape
        tf_pose_est = self.get_correct_pose_estimator(img_w, img_h)
        
        # Downscale the image beforehand will increase the execution time, slightly.
        image_original_bgr = cv2.resize(image_original_bgr, (self.scales[0][0]*16, self.scales[0][1]*16))

        skeletons = tf_pose_est.inference(image_original_bgr, upsample_size=4.0)

        # Transform its location and determine the correct skeleton given the head position.
        margin_top, margin_bottom, margin_left, margin_right = patch_margins
        face_top, face_bottom, face_left, face_right = face_position
        face_position = (face_top-margin_top, face_bottom-margin_top, face_left-margin_left, face_right-margin_left)
        correct_skeleton = self.get_corresponding_skeleton(skeletons, face_position, img_h, img_w)

        if correct_skeleton is None:
            print("No skeleton in the face position found")
            return False
        
        image_drawn = tf_pose_est.draw_humans(image_drawn, [correct_skeleton], imgcopy=False)
        
        self.lstm_gesture_recognition.append_skeleton(correct_skeleton)
        label = self.lstm_gesture_recognition.predict()

        self.third_last = self.second_last
        self.second_last = self.first_last
        self.first_last = label
        
        # labels 0-4 are activated by: hand up, hand down, half circle up, half circle down and noise.
        if self.third_last == self.second_last and self.second_last == self.first_last:
            if label == 0:
                if self.droneController.perform_action:
                    self.droneController.perform_action(Action.MOVE_UP)
            elif label == 1:
                if self.droneController.is_online:
                    self.droneController.perform_action(Action.MOVE_DOWN)
            elif label == 2 or label == 3:
                # If half circle up or half circle down, do flip. For savety, after one flip, disable flipping.
                if self.droneController.allow_flip:
                    if self.droneController.perform_action:
                        self.droneController.perform_action(Action.FLIP)
                    self.Allow_flip = False
                else:
                    print("Flip not allowed!")
                
            # After a action has been performed, initialize the queue with the noise class.
            self.first_last = 4
            self.second_last = 4
            self.third_last = 4

        return True