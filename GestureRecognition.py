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
        loader = tf.train.import_meta_graph("./lstm_model_cossin/model.meta")
        loader.restore(self.sess, tf.train.latest_checkpoint("./lstm_model_cossin"))
          
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
         
    def get_angle(p1, p2, p3, side):
        radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
        if side == 'left':
            return -math.sin(radians), math.cos(radians)
        else:
            return math.sin(radians), math.cos(radians)
    
    def get_angles(self, skeleton):
        if all(part in skeleton.body_parts for part in [1, 2, 3, 4, 5, 6, 7]):
                        
            neck = skeleton.body_parts[1]
            rshoulder = skeleton.body_parts[2]
            relbow = skeleton.body_parts[3]
            rwrist = skeleton.body_parts[4]
            lshoulder = skeleton.body_parts[5]
            lelbow = skeleton.body_parts[6]
            lwrist = skeleton.body_parts[7]
    
            angle_lelbow_sin, angle_lelbow_cos = self.get_angle(lwrist, lelbow, lshoulder, "left")
            angle_relbow_sin, angle_relbow_cos = self.get_angle(rwrist, relbow, rshoulder, "right")
            angle_lshoulder_sin, angle_lshoulder_cos = self.get_angle(lelbow, lshoulder, neck, "left")
            angle_rshoulder_sin, angle_rshoulder_cos = self.get_angle(relbow, rshoulder, neck, "right")
            
        
            return [angle_relbow_sin, angle_relbow_cos, angle_rshoulder_sin, angle_rshoulder_cos, angle_lshoulder_sin, angle_lshoulder_cos, angle_lelbow_sin, angle_lelbow_cos]
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
#        self.positional_recognition = PositionalRecognition(droneController)

    def get_correct_pose_estimator(self, w, h):
        ratio = w/h
        idx = (np.abs(self.pose_estimators_ratios - ratio)).argmin()
        
        
#        print("Ratio {} -> {} ({})".format(ratio, self.pose_estimators_ratios[idx], idx))
        
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
        
        image_drawn = tf_pose_est.draw_humans(image_drawn, [correct_skeleton], imgcopy=False)
        
#        self.positional_recognition.perform_action(image_original.shape, patch_margins, correct_skeleton)
#        self.lstm_gesture_recognition.append_skeleton(correct_skeleton)
#        label = self.lstm_gesture_recognition.predict()
#        
#        if label == 0:
#            print("Waving left hand")
#        elif label == 1:
#            print("Waving right hand")
#        elif label == 2:
#            print("Clapping")
#        else:
#            print("Other action")

        return True