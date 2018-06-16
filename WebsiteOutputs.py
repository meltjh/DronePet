import time
import matplotlib.pyplot as plt
import cv2


                
class WebsiteOutput:
    communication = None
    
    def __init__(self, communication):
        print('WebsiteOutput')
        self.communication = communication
        
        #cv2.namedWindow('img', cv2.CV_WINDOW_AUTOSIZE)
        cv2.startWindowThread()
    
    def show_output(self):
        print('show_output')
        while self.communication.active == True:
            time.sleep(1)
            
            if self.communication.last_image_processed is not None:
                plt.imshow(self.communication.last_image_processed)
                plt.show()
                
#                print('aaaa')
#                cv2.startWindowThread()
#                cv2.imshow('img', self.communication.last_image_original)
#                cv2.waitKey(1000)
#                print('bbbb')