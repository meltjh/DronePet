from threading import Thread
import cv2
import matplotlib.pyplot as plt

class OfflineDroneController:
    
    def __init__(self, args):
       print("OfflineDroneController") 
        
        
    def perform_action(self, command, value = None):
        print("perform_action", command, value)
        
class OfflineBebop:
    def __init__(self):
        print("OfflineBebop")
    
    def start_video_stream(self):
        print("start_video_stream")
        
        
class OfflineDroneVisionGUI:
    
    def __init__(self, bebop, is_bebop, user_args):
        print("OfflineDroneVisionGUI")
#        self.cap = cv2.VideoCapture(0)
        img = cv2.imread("face_recognition_data/faces.jpg")
        img = cv2.resize(img, (1280, 720))
#        img = cv2.resize(img, (640, 480))
        self.img = img
        
    def get_latest_valid_picture(self):
        print("get_latest_valid_picture")
#        _, img = self.cap.read()
        return self.img