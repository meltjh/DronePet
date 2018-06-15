import cv2
import numpy as np
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from FaceRecognition import FaceRecognition

class GestureInput:
    communication = None
    vision = None
    faceRecognition = None
    face_id = None
    
    def __init__(self, communication, vision):
        print('GestureInput')
        self.communication = communication
        self.vision = vision
        
        self.faceRecognition = FaceRecognition()
        
        self.h = 432
        self.w = 368
        self.poseEstimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(self.h,self.w))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.face_id = 1
     
#    def face_training(self):
        
    def face_recognition(self, image_original, image_drawn, specific_face_id):
        
        # Recogize faces
        faces = self.faceRecognition.recognize_faces(image_original)
        if len(faces) <= 0:
            print('No faces are found.')
            return None, None
        
        # Draw the faces and select the first face that has the given id.
        correct_face = None
        for face_id, face in faces:
            if face_id == specific_face_id and correct_face is None:
                # Set the found face as the correct face. This is only the first instance.
                correct_face = face
                # Draw the correct face green
                image_drawn = self.faceRecognition.draw_face(image_drawn, face, face_id)
            else:
                # Draw the incorrect faces red
                image_drawn = self.faceRecognition.draw_face(image_drawn, face, face_id, box_color = (192, 0, 0))
        
        if correct_face is None:
            print('The correct face is not found.')
            return None, image_drawn
        
        # Determine the body patch corners
        top, bottom, left, right = self.faceRecognition.body_patch(image_original, correct_face)
        # Obtain the patch
        image_patch = image_original[top:bottom, left:right, :]
        # Draw the rectangle on the normal image
        cv2.rectangle(image_drawn, (left, top), (right, bottom), (0, 255, 0), 4)
        
        return image_patch, image_drawn


        
#    def gesture_recognition(self):
        
    
        
        
    def processing_stream(self, args):
        print('processing_stream')
        
        while self.communication.active == True:
            # Obtain a image from the stream
            image_original = self.vision.get_latest_valid_picture()
            
            if image_original is None:
                print('No image found')
                self.communication.last_image = None
                return
            
            image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        
            # Face recognition
            image_patch, image_drawn = self.face_recognition(image_original=image_original, image_drawn=image_original, specific_face_id=self.face_id)
        
            self.communication.last_image = image_drawn
        
            # Skeleton recognition
        
        


            
            
#            img_to_save = None
#            patch_with_skeleton = None
#            
#            if img is not None:
#                img_to_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                face_x, face_y, face_w, face_h = self.faceRecognition.recognize_face(img_to_save, self.face_id)
#            
#                if min(face_x, face_y, face_w, face_h) >= 0:
#                
#                    # Patch pakken
#                    [img_height, img_width, _] = img_to_save.shape
#                    
#                    # TODO: goede margins bedenken
#                    margin_width_ratio = 3
#                    margin_top_ratio = 1
#                    margin_bottom_ratio = 10
#                    
#                    margin_width = int(round(margin_width_ratio * face_w))
#                    margin_top = int(round(margin_top_ratio) * face_h)
#                    margin_bottom = int(round(margin_bottom_ratio) * face_h)
#                    
#                    # TODO margins boven hoofd en lengte ook ipv einde plaatje?
#                    left = max(0, face_x - margin_width)
#                    right = min(img_width, face_x + face_w + margin_width)
#                    
#                    top = max(0, face_y - margin_top)
#                    bottom = min(img_height, face_y + face_h + margin_bottom)
#                    
#                    patch = img_to_save[top:bottom, left:right, :]
#                    
##                    self.communication.last_image = patch
#                    
#                    # Skeleton
#                    humans = self.poseEstimator.inference(patch, upsample_size=4.0)
#                    if len(humans) >= 1: # TODO wat met meerdere humans?
#                        patch_with_skeleton = TfPoseEstimator.draw_humans(patch, humans, imgcopy=False)
#                        # Put patch with skeleton on the original image, only for visual purpose.
#                        img_to_save[top:bottom, left:right, :] = patch_with_skeleton                        
#                    
#                        # Rectangle op gezicht plakken + id
#                        
#                        
#                        if self.face_id == 0:
#                            name = 'Richard'
#                        if self.face_id == 1:
#                            name = 'Melissa'
#                        else:
#                            name = 'Unknown'
#                        cv2.putText(img_to_save, str(name), (face_x, face_y - 40), self.font, 1, (255, 255, 255), 3)
#                    else:
#                        img_to_save = None
#                        print("Not detected")
#
#            self.communication.last_image = patch_with_skeleton


