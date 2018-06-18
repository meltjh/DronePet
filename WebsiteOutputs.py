import time
import matplotlib.pyplot as plt
import cv2


                
class WebsiteOutput:
    communication = None
    
    def __init__(self, communication):
        print('WebsiteOutput')
        self.communication = communication
        
        cv2.namedWindow('Frame')
        cv2.startWindowThread()
    
    def show_output(self):
        print('show_output')
        while self.communication.active == True:            
            if self.communication.last_image_processed is not None:
                frame_resized = cv2.resize(self.communication.last_image_processed, (0,0), fx=0.5, fy=0.5)
                cv2.imshow('Frame', cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        cv2.destroyAllWindows()
