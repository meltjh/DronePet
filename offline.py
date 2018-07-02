from threading import Thread
import cv2
import time
import sys

from actions import Action

class OfflineDroneController:
    
    def __init__(self, args):
       print("Initializing the offline drone controller") 
       self.bebop = OfflineBebop()
       self.allow_flip = False
       self.is_online = False
       
    def perform_action(self, cmd, value = None):
        print("Performing offline action:", cmd, value)
        
        
        if cmd == Action.ALLOW_FLIP:
            self.allow_flip = True
            return
        if cmd == Action.DISALLOW_FLIP:
            self.allow_flip = False
            return
        
        
class OfflineBebop:
    IsOnlineBebop = False
    
    def __init__(self):
        print("Initializing offline bebop")
    
    def start_video_stream(self):
        print("Starting the offline video stream")
        
        
class OfflineDroneVisionGUI:
    
    def __init__(self, bebop, is_bebop, user_args):
        print("Initializing the offline drone vision gui")
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        
        # The code below is to test the stream with images instead of the webcam.
#        self.counter = 1
        
    def get_latest_valid_picture(self):
        # if something is wrong in this code, it is hard to end this process without the sleep.
        time.sleep(0.01)
        _, img = self.cap.read()
        
        # To test with individual images.
#        if self.counter > some_max_i:
#            sys.exit("End of image stream, ending program")
#        img = cv2.imread("test_img{}.jpg".format(self.counter))
#        self.counter+=1
        return img