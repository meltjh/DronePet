import cv2
import numpy as np
import matplotlib.pyplot as plt
from FaceRecognition import FaceRecognition
from GestureRecognition import GestureRecognition
from Actions import Action
import traceback
from enum import Enum

class Training(Enum):
    NoTraining = 0
    IdTraining = 1
    AllTraining = 2



    
class StreamInput:
    communication = None
    vision = None
    faceRecognition = None
    face_id = None
    
    def __init__(self, communication, vision):
        print('GestureInput')
        self.communication = communication
        self.vision = vision
        
        self.faceRecognition = FaceRecognition()
        self.gestureRecognition = GestureRecognition(self.faceRecognition, self.communication)
        
        # If true, save the patches of the face found by the id
        self.face_id = 1
        self.training = Training.NoTraining
        
    def processing_stream(self, args):
        print('processing_stream')
        
        while self.communication.active == True:
            try:
                # Obtain a image from the stream
                image_original = self.vision.get_latest_valid_picture()
                
                if image_original is None:
                    print('No image found')
                    self.communication.last_image_processed = None
                    self.communication.last_image_original = None
                    return
                
                image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
                
                self.communication.last_image_original = image_original
                image_drawn = image_original.copy()
                [image_h, image_w, _] = image_original.shape
                
                
                if self.training == Training.AllTraining:
                    self.faceRecognition.store_training_faces(image_original, self.face_id)
                    return
                        
                
                # Detect the most confident face given the face_id
                # Show all faces and get the location in which the human should be.
                (top, bottom, left, right) = self.faceRecognition.main(image_original, image_drawn, self.face_id)
                

                if max(left, top, right, bottom) < 0:
                    print('No patch found by face_recognition')
                    self.communication.last_image_processed = image_drawn
                    return

                # Crop the images to feed to the skeleton recognition
                image_original_patch = image_original[top:bottom, left:right, :]
                image_drawn_patch = image_drawn[top:bottom, left:right, :]

                # Skeleton recognition
                patch_margins = (top, image_h-bottom, left, image_w-right)
                face_top, face_bottom, face_left, face_right= self.gestureRecognition.main(image_original=image_original_patch, image_drawn=image_drawn_patch, specific_face_id=self.face_id, patch_margins=patch_margins)
                
                # Combine the patch with the full image
                image_drawn[top:bottom, left:right, :] = image_drawn_patch
                self.communication.last_image_processed = image_drawn

            
                # Save the face patch that was found by gestureRecognition
                if self.training == Training.IdTraining:
                    if image_original_patch is not None and min(face_left, face_top, face_right, face_bottom) >= 0:
                        image_patch_face = image_original_patch[face_top:face_bottom, face_left:face_right, :]
                        self.faceRecognition.store_training_face(image_patch_face, self.face_id)
                    
            except Exception as e:
                print("Error in Processin_stream, type error: " + str(e))
                print(traceback.format_exc())
                
                
                