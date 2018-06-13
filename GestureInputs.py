class GestureInput:
    communication = None
    vision = None
    
    def __init__(self, communication, vision):
        print('GestureInput')
        self.communication = communication
        self.vision = vision
        
    def obtain_image(self, args):
        print('obtain_image')
        while self.communication.active == True:
            
            img = self.vision.get_latest_valid_picture()
            # TODO: hele gesture_input gedeelte
            
            self.communication.last_image = img