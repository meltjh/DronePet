from threading import Thread
import cv2
import time
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
        self.cap = cv2.VideoCapture(0)
        
    def get_latest_valid_picture(self):
        print("get_latest_valid_picture")
        time.sleep(1)
        _, img = self.cap.read()
        return img