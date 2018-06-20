import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from Actions import Action
from enum import Enum

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
    
class GestureRecognition:
    
    def __init__(self, faceRecognition, droneController):
        
        # Detection parameters
        self.h = 352 #640
        self.w = 352 #480
        # Load the trained mode
        self.poseEstimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(self.h,self.w))
    
        # Face recognition is used in id_skeleton
        self.faceRecognition = faceRecognition
        self.droneController = droneController
            
            
            
    def determine_camera_movements(self, skeleton, patch_shape, patch_margins):

        [margin_top, margin_bottom, margin_left, margin_right] =  patch_margins   
        # If there was some margin, calculate the relative positions and update the skeleton.
        if max(margin_top, margin_bottom, margin_left, margin_right) > 0:
            
            [patch_h, patch_w, _] = patch_shape
            offset_x = margin_left
            offset_y = margin_top
            full_w = margin_left + patch_w + margin_right
            full_h = margin_top + patch_h + margin_bottom
        
            for i in skeleton.body_parts.keys():
                skeleton.body_parts[i].x =  (skeleton.body_parts[i].x*patch_w+offset_x)/full_w
                skeleton.body_parts[i].y =  (skeleton.body_parts[i].y*patch_h+offset_y)/full_h
        
        nose = skeleton.body_parts[0]
        
        nose_x_scale = nose.x
        
        
        print("NOSE SCALE X", nose_x_scale)
        middle = 0.5
        margin = 0.05
        
        command = Action.NOTHING
        if nose_x_scale > middle + margin:
#            offset = nose_x_scale - (middle + margin)
            offset = nose_x_scale - middle
            command = Action.LOOK_RIGHT
            print("GO TO RIGHT")
        elif nose_x_scale < middle - margin:
#            offset = (middle - margin) - nose_x_scale
            offset = middle  - nose_x_scale
            command = Action.LOOK_LEFT
            print("GO TO LEFT")
        else:
            offset = 0
        
        if offset != 0:
            print("OFFSET X" , offset)
            value = (((offset+1)**11)-1)
            self.droneController.perform_action(command, value)
    
    
    
        nose_y_scale = nose.y
        
        
        print("NOSE SCALE Y", nose_y_scale)
        middle = 0.5 # <.5 so see more body
        margin = 0.05
        
        command = Action.NOTHING
        if nose_y_scale > middle + margin:
#            offset = nose_y_scale - (middle + margin)
            offset = nose_y_scale - middle
            command = Action.LOOK_DOWN
            print("GO TO DOWN")
        elif nose_y_scale < middle - margin:
#            offset = (middle - margin) - nose_y_scale
            offset = middle - nose_y_scale
            command = Action.LOOK_UP
            print("GO TO UP")
        else:
            offset = 0
        
        if offset != 0:
            print("OFFSET Y" , offset)
            value = (((offset+1)**9)-1)
            self.droneController.perform_action(command, value)
            
    # Given the eyes, nose and if possible the ears, estimate the face patch
    # which can then be fed to face recognition to determine the id of the skeleton.
    # Returns the (top, bottom, left, right) pixel values of the patch.
    # (-1, -1, -1, -1) if no patch found.
    def get_face_patch(self, image, skeleton):
        # If no nose and eyes found, continue.
        if not (0 in skeleton.body_parts and
                14 in skeleton.body_parts and
                15 in skeleton.body_parts):
            
            print('No eyes and nose found')
            return (-1, -1, -1, -1)
        
        
        [image_h, image_w, _] = image.shape
        
        nose = skeleton.body_parts[0]
        eye_r = skeleton.body_parts[15] # l and r switched to relative to camera
        eye_l = skeleton.body_parts[14] # l and r switched to relative to camera
        
        cv2.circle(image, (int(round(image_w*eye_r.x)), int(round(image_h*eye_r.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
        cv2.circle(image, (int(round(image_w*eye_l.x)), int(round(image_h*eye_l.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
        cv2.circle(image, (int(round(image_w*nose.x)), int(round(image_h*nose.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)


        best_right = eye_r.x + (eye_r.x - nose.x)
        if 17 in skeleton.body_parts:
            ear_r = skeleton.body_parts[17] # l and r switched to relative to camera
            
            cv2.circle(image, (int(round(image_w*ear_r.x)), int(round(image_h*ear_r.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
            
            
            best_right = ear_r.x
        
        best_left = eye_l.x - (nose.x - eye_l.x)
        if 16 in skeleton.body_parts:
            ear_l = skeleton.body_parts[16] # l and r switched to relative to camera
            
            cv2.circle(image, (int(round(image_w*ear_l.x)), int(round(image_h*ear_l.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
                 
            best_left = ear_l.x
        
        face_width = best_right - best_left
        
        # Crop the head, and add a little bit the difference between eyes and nose.
        # When this increases, the face is in front of the camera, so the head seems biggest.
        # When the head is seen from above or below, the head is smaller, and so is the difference between the eyes and the nose.
        avg_eyes = (eye_r.y + eye_l.y) / 2
        nose_eyes_difference = nose.y - avg_eyes
        face_height = face_width*2 + nose_eyes_difference
        
        left = best_left
        top = nose.y - (face_height / 2) * 1.2
        right = best_right
        bottom = nose.y + (face_height / 2)
        
        
        
        
        # Round and normalize to the actual image pixels
        left_inner = int(round(left*image_w))
        top_inner = int(round(top*image_h))
        right_inner = int(round(right*image_w))
        bottom_inner = int(round(bottom*image_h))
        
        self.faceRecognition.draw_box(image, (top_inner, bottom_inner, left_inner, right_inner), title = 'Face inner', box_color = (255, 255, 255))


        width = right-left
        height = bottom - top

        FACE_MARGIN_WIDTH = 0.5
        FACE_MARGIN_HEIGHT = 0.4
        # Add some margin:
        left = left - FACE_MARGIN_WIDTH * width
        top = top - FACE_MARGIN_HEIGHT * height 
        right = right + FACE_MARGIN_WIDTH * width
        bottom = bottom + FACE_MARGIN_HEIGHT * height
        

        # Round and normalize to the actual image pixels
        left = int(round(left*image_w))
        top = int(round(top*image_h))
        right = int(round(right*image_w))
        bottom = int(round(bottom*image_h))
        
        left = max(0, left)
        top = max(0, top)
        right = min(image_w, right)
        bottom = min(image_h, bottom)
        
        self.faceRecognition.draw_box(image, (top, bottom, left, right), title = 'Face margin', box_color = (255, 255, 255))
        
        return top, bottom, left, right

    def id_skeletons(self, image_original, image_drawn, skeletons):
        skeletons_by_ids = []
        for skeleton in skeletons: 
            top, bottom, left, right = self.get_face_patch(image_drawn, skeleton)
            
            # Obtain the patch
            image_patch = image_original[top:bottom, left:right, :]

            # Recogize faces
            faces = self.faceRecognition.recognize_faces(image_patch)
            for face_id, (s_top, s_bottom, s_left, s_right), confidence in faces:
                # Add offset
                corners = (s_top + top, s_bottom + top, s_left + left, s_right + left)
                skeletons_by_ids.append((skeleton, face_id, corners, confidence))
                
        # Sort faces so that the most confident are in front of the list.
        skeletons_by_ids.sort(key=lambda x: x[3], reverse = True)

        return skeletons_by_ids
    
    
    # Draws the skeleton and returns the exact relative face position of the given id, or if None, the first face.
    def main(self, image_original, image_drawn, specific_face_id = None, patch_margins = (0,0,0,0)):
        if image_original is None:
            print('No image given for gesture_recognition')
            return (-1, -1, -1, -1)
        
        # Obtain the skeletons
        image_original_bgr = cv2.cvtColor(image_original[:,:,:], cv2.COLOR_RGB2BGR)
        skeletons = self.poseEstimator.inference(image_original_bgr, upsample_size=4.0)
        
        if len(skeletons) <= 0:
            print("Could not find skeleton(s)")
            return (-1, -1, -1, -1)
        
        print("\n\nskeletons printen")
        skeletons_by_ids = self.id_skeletons(image_original, image_drawn, skeletons)

        for skeleton, face_id, (face_top, face_bottom, face_left, face_right), _ in skeletons_by_ids:
            if specific_face_id is None or face_id == specific_face_id:
                name = self.faceRecognition.id_to_name(specific_face_id)
                correct_skeleton = skeleton
                
                image_drawn = self.faceRecognition.draw_box(image_drawn, (face_top, face_bottom, face_left, face_right), "skeleton_id "+name, box_color = (100, 100, 100))
                
                self.determine_camera_movements(correct_skeleton, image_original.shape, patch_margins)
                
                # The first one will be the best one since it is sorted by confidence
                image_drawn = TfPoseEstimator.draw_humans(image_drawn, [correct_skeleton], imgcopy=False)
                
                return (face_top, face_bottom, face_left, face_right)
                
        return (-1, -1, -1, -1)
        