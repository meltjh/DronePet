import cv2
import numpy as np
import traceback

import matplotlib.pyplot as plt
from face_recognition_2 import FaceRecognition
from gesture_recognition import GestureRecognition
from actions import Action

import time


    
class StreamInput:
    
    def __init__(self, vision, streamOutput, droneController):
        print('GestureInput')
        self.droneController = droneController
        self.vision = vision
        
        self.faceRecognition = FaceRecognition(droneController)
        self.gestureRecognition = GestureRecognition(self.faceRecognition, self.droneController)
        self.streamOutput = streamOutput
        
        # The face that it should detect and 'listen' to.
        self.face_id = 0
        
    def processing_stream(self, args):
        # Infinite loop to get the latest valid picture. It will stop when one is found.
        while True:
            try:
                # Obtain a image from the stream
                image_original = self.vision.get_latest_valid_picture()
                
                if image_original is None:
                    print('No image found')
                    self.streamOutput.update_stream(None)
                    return
                
                # The input is BGR instead of RGB, so transform this to RGB.
                image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
                image_drawn = image_original.copy()
                [image_h, image_w, _] = image_original.shape                        
                
                
                start = time.clock()
                # Obtain the body and face patch locations.
                (body_top, body_bottom, body_left, body_right), (face_top, face_bottom, face_left, face_right) = self.faceRecognition.main(image_original, image_drawn, self.face_id)
                end = time.clock()
                print('1x face detection time {} seconds'.format(round(end - start,3)))
            
            
            
                if max(body_top, body_bottom, body_left, body_right) < 0:
                    print('No patch found by face_recognition')
                    self.streamOutput.update_stream(image_drawn)
                    return

                # Crop the images to feed to the skeleton recognition
                image_original_patch = image_original[body_top:body_bottom, body_left:body_right, :]
                image_drawn_patch = image_drawn[body_top:body_bottom, body_left:body_right, :]

                # Skeleton recognition
                # The patch margins are used to determine the location of the patch
                patch_margins = (body_top, image_h-body_bottom, body_left, image_w-body_right)
                _ = self.gestureRecognition.main(image_original_patch, image_drawn_patch, (face_top, face_bottom, face_left, face_right), self.face_id, patch_margins)
                
                # Combine the patch with the full image
                image_drawn[body_top:body_bottom, body_left:body_right, :] = image_drawn_patch
                self.streamOutput.update_stream(image_drawn)

            except Exception as e:
                print("Error in Processin_stream, type error: " + str(e))
                print(traceback.format_exc())  