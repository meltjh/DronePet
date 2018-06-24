import cv2
import numpy as np
import matplotlib.pyplot as plt
from FaceRecognition import FaceRecognition
from GestureRecognition import GestureRecognition
from Actions import Action
import traceback
from enum import Enum
import sys

class Training(Enum):
    NoTraining = 0
    IdTraining = 1
    AllTraining = 2
    
class StreamInput:
    communication = None
    vision = None
    faceRecognition = None
    face_id = None
    
    def __init__(self, vision, streamOutput, droneController):
        print('GestureInput')
        self.droneController = droneController
        self.vision = vision
        
        self.faceRecognition = FaceRecognition()
        self.gestureRecognition = GestureRecognition(self.faceRecognition, self.droneController)
        self.streamOutput = streamOutput
        
        # If true, save the patches of the face found by the id
        self.face_id = 0
        self.training = Training.AllTraining
        self.train_only = True # Only do face recognition
        
    def processing_stream(self, args):
        print('processing_stream')
        
        while True:
            try:
                # Obtain a image from the stream
                image_original = self.vision.get_latest_valid_picture()
                
                if image_original is None:
                    print('No image found')
                    self.streamOutput.update_stream(None)
                    return
                
                image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
                image_drawn = image_original.copy()
                [image_h, image_w, _] = image_original.shape
                
                
                if self.training == Training.AllTraining:
                    self.faceRecognition.store_training_faces(image_original, self.face_id, True, image_drawn)                    
                    
                    if self.train_only:
                        self.streamOutput.update_stream(image_drawn)
                        sys.exit("Done testing")
                        return
                        
                
                # Detect the most confident face given the face_id
                # Show all faces and get the location in which the human should be.
                (body_top, body_bottom, body_left, body_right), (face_top, face_bottom, face_left, face_right) = self.faceRecognition.main(image_original, image_drawn, self.face_id)

                if max(body_top, body_bottom, body_left, body_right) < 0:
                    print('No patch found by face_recognition')
                    self.streamOutput.update_stream(image_drawn)
                    return
                
                # Save the face patch that was found by gestureRecognition
                if self.training == Training.IdTraining:
                    image_patch_face = image_original[face_top:face_bottom, face_left:face_right, :]
                    self.faceRecognition.store_training_face(image_patch_face, self.face_id)
                    
                    if self.train_only:
                        self.streamOutput.update_stream(image_drawn)
                        return

                # Crop the images to feed to the skeleton recognition
                image_original_patch = image_original[body_top:body_bottom, body_left:body_right, :]
                image_drawn_patch = image_drawn[body_top:body_bottom, body_left:body_right, :]

                # Skeleton recognition
                patch_margins = (body_top, image_h-body_bottom, body_left, image_w-body_right)
                succes = self.gestureRecognition.main(image_original=image_original_patch, image_drawn=image_drawn_patch, face_position=(face_top, face_bottom, face_left, face_right), specific_face_id=self.face_id, patch_margins=patch_margins)
                
                # Combine the patch with the full image
                image_drawn[body_top:body_bottom, body_left:body_right, :] = image_drawn_patch
                self.streamOutput.update_stream(image_drawn)
                
            except Exception as e:
                print("Error in Processin_stream, type error: " + str(e))
                print(traceback.format_exc())
                
                
                