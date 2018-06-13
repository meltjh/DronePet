import time
import matplotlib.pyplot as plt


class WebsiteOutput:
    communication = None
    
    def __init__(self, communication):
        print('WebsiteOutput')
        self.communication = communication
    
    def show_output(self):
        print('show_output')
        while self.communication.active == True:
            time.sleep(1)
            
            if self.communication.last_image is not None:
                plt.imshow(self.communication.last_image)
                plt.show()