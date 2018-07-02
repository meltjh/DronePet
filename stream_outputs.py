import matplotlib.pyplot as plt
import cv2

# When true, a stream will be shown rather than individual images in the console.
# The stream might not work in some environments.
is_using_stream = False
                
class StreamOutput:
    
    def __init__(self):
        if is_using_stream:
            cv2.namedWindow('StreamOutput')
            cv2.startWindowThread()
    
    def update_stream(self, frame):    
        if frame is not None:
            # Downsizing the resolution of the stream, optional.
            frame_resized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            
            if is_using_stream:
                cv2.imshow('StreamOutput', cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(frame_resized)
                plt.show()
