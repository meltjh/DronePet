from Actions import Action

class Communication:
    active = True
    last_command = Action.NOTHING
    last_image_original = None
    last_image_processed = None
    
    def __init__(self):
        print("Communication class is initialized")

#        self.last_command = -1
#        self.last_image = None