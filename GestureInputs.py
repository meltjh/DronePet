import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        
        self.face_id = 1
        
    def obtain_image(self, args):
        print('obtain_image')
        while self.communication.active == True:
            
            img = self.vision.get_latest_valid_picture()
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, x, y, w, h = self.faceRecognition.recognize_face(img, self.face_id)
            
                # TODO: hele gesture_input gedeelte
            
            self.communication.last_image = img


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
            
#        plt.imshow(image)
#        plt.show()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        # Get all face from the video frame
        faces = self.faceCascade.detectMultiScale(gray, 1.2,5)
    
        # For each face in faces
        for(x,y,w,h) in faces:
    
            # Create rectangle around the face
            cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
    
            # Recognize the face belongs to which ID
            Id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
    
            # Weet niet waarom ze dat hebben gedaan
            confidence = round(100 - confidence, 2)
            
            if confidence < 0.25:
                continue
            
            if Id != face_id:
                continue
            
            if(Id == 0):
                Id = "Rica {0:.2f}%".format(round(confidence, 2))
                
            if(Id == 1):
                Id = "Mel {0:.2f}%".format(round(confidence, 2))
    
            if(Id == 2):
                Id = "Ms Box {0:.2f}%".format(round(confidence, 2))
                
                
            # Put text describe who is in the picture
            cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            cv2.putText(image, str(Id), (x,y-40), self.font, 1, (255,255,255), 3)
            
            
            return image, x, y, w, h
        return image, -1, -1, -1, -1