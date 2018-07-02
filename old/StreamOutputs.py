import matplotlib.pyplot as plt
import cv2

PYTHONW = True
                
class StreamOutput:
    
    def __init__(self):
        print('WebsiteOutput')
        
        if PYTHONW:
            cv2.namedWindow('StreamOutput')
            cv2.startWindowThread()
    
    def update_stream(self, frame):    
        if frame is not None:
            frame_resized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            
            if PYTHONW:
                cv2.imshow('StreamOutput', cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(frame_resized)
                plt.show()
#        cv2.destroyAllWindows()
