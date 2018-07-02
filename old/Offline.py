from threading import Thread
import cv2
import matplotlib.pyplot as plt
import sys
import time
from Actions import Action

class OfflineDroneController:
    
    def __init__(self, args):
       print("OfflineDroneController") 
       self.bebop = OfflineBebop()
        
       self.allow_flip = False
       
       
    def perform_action(self, cmd, value = None):
        print("perform_action", cmd, value)
        
        
        if cmd == Action.ALLOW_MOVEMENTS:
            print("ALLOW_MOVEMENTS")
            self.allow_flip = True
            return
        if cmd == Action.DISALLOW_MOVEMENTS:
            print("DISALLOW_MOVEMENTS")
            self.allow_flip = False
            return
        
        
class OfflineBebop:
    IsOnlineBebop = False
    
    def __init__(self):
        print("OfflineBebop")
    
    def start_video_stream(self):
        print("start_video_stream")
        
        
class OfflineDroneVisionGUI:
    
    def __init__(self, bebop, is_bebop, user_args):
        print("OfflineDroneVisionGUI")
        self.cap = cv2.VideoCapture(0)
        
        self.counter = 1
#        img = cv2.imread("face_recognition_data/faces.jpg".format(counter))
#        img = cv2.imread("posture.jpg")
#        img = cv2.resize(img, (1280, 720))
#        img = cv2.resize(img, (640, 480))
#        self.img = img
        
    def get_latest_valid_picture(self):
#        print("get_latest_valid_picture")
        time.sleep(0.01)
        _, img = self.cap.read()
#        img = img = cv2.imread("face_recognition_data/faces_by_hand/User.0.10.jpg".format(self.counter))
#        img = img = cv2.imread("face_recognition_data/faces_by_hand/User.1.{}.jpg".format(self.counter))
#        self.counter+=1
#        if self.counter > 12:
#            sys.exit("End of testing")
#        img = cv2.imread("posture.jpg")
        return img