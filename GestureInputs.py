import cv2
import numpy as np
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

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
        
    def obtain_image(self, args):
        print('obtain_image')
        while self.communication.active == True:
            
            img = self.vision.get_latest_valid_picture()
            img_to_save = None
            patch_with_skeleton = None
            
            if img is not None:
                img_to_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_x, face_y, face_w, face_h = self.faceRecognition.recognize_face(img_to_save, self.face_id)
            
                if min(face_x, face_y, face_w, face_h) >= 0:
                
                    # Patch pakken
                    [img_height, img_width, _] = img_to_save.shape
                    
                    # TODO: goede margins bedenken
                    margin_width_ratio = 3
                    margin_top_ratio = 1
                    margin_bottom_ratio = 10
                    
                    margin_width = int(round(margin_width_ratio * face_w))
                    margin_top = int(round(margin_top_ratio) * face_h)
                    margin_bottom = int(round(margin_bottom_ratio) * face_h)
                    
                    # TODO margins boven hoofd en lengte ook ipv einde plaatje?
                    left = max(0, face_x - margin_width)
                    right = min(img_width, face_x + face_w + margin_width)
                    
                    top = max(0, face_y - margin_top)
                    bottom = min(img_height, face_y + face_h + margin_bottom)
                    
                    patch = img_to_save[top:bottom, left:right, :]
                    
#                    self.communication.last_image = patch
                    
                    # Skeleton
                    humans = self.poseEstimator.inference(patch, upsample_size=4.0)
                    if len(humans) >= 1: # TODO wat met meerdere humans?
                        patch_with_skeleton = TfPoseEstimator.draw_humans(patch, humans, imgcopy=False)
                        # Put patch with skeleton on the original image, only for visual purpose.
                        img_to_save[top:bottom, left:right, :] = patch_with_skeleton                        
                    
                        # Rectangle op gezicht plakken + id
                        cv2.rectangle(img_to_save, (face_x - 20, face_y - 20), (face_x + face_w + 20, face_y + face_h + 20), (0, 255, 0), 4)
                        if self.face_id == 0:
                            name = 'Richard'
                        if self.face_id == 1:
                            name = 'Melissa'
                        else:
                            name = 'Unknown'
                        cv2.putText(img_to_save, str(name), (face_x, face_y - 40), self.font, 1, (255, 255, 255), 3)
                    else:
                        img_to_save = None
                        print("Not detected")

            self.communication.last_image = patch_with_skeleton


# Modified by Nazmi Asri
# Original code: http://thecodacus.com/ 
class FaceRecognition:
    
    def __init__(self):
        # Create Local Binary Patterns Histograms for face recognization
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load the trained mode
        self.recognizer.read('face_recognition_data/trainer/trainer.yml')
        
        # Load prebuilt model for Frontal Face
        cascadePath = "face_recognition_data/haarcascade_frontalface_default.xml"
        
        # Create classifier from prebuilt model
        self.faceCascade = cv2.CascadeClassifier(cascadePath);
        
        # Set the font style
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def recognize_face(self, image, face_id):
        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        # Get all face from the video frame
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
        # For each face in faces
        for(x,y,w,h) in faces:
    
            # Create rectangle around the face
#             cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
    
            # Recognize the face belongs to which ID
            Id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
    
            # Weet niet waarom ze dat hebben gedaan
            confidence = round(100 - confidence, 2)
            
            if confidence < 0.25:
                continue
            
            if Id != face_id:
                continue
            
#            if(Id == 0):
#                Id = "Rica {0:.2f}%".format(round(confidence, 2))
#                
#            if(Id == 1):
#                Id = "Mel {0:.2f}%".format(round(confidence, 2))
#    
#            if(Id == 2):
#                Id = "Ms Box {0:.2f}%".format(round(confidence, 2))
                
                
            # Put text describe who is in the picture
#            cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
#            cv2.putText(image, str(Id), (x,y-40), self.font, 1, (255,255,255), 3)
#            
            
            return x, y, w, h
        return -1, -1, -1, -1