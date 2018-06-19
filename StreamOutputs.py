import matplotlib.pyplot as plt
import cv2


                
class StreamOutput:
    
    def __init__(self):
        print('WebsiteOutput')
        
        cv2.namedWindow('StreamOutput')
        cv2.startWindowThread()
    
    def update_stream(self, frame):    
        if frame is not None:
            frame_resized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('StreamOutput', cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
#        cv2.destroyAllWindows()
