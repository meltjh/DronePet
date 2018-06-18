from Actions import Action

class Communication:
    active = True
    last_image_original = None
    last_image_processed = None
    droneController = None
    
    def __init__(self, droneController):
        print("Communication class is initialized")

        self.droneController = droneController
        
        
    def send_command(self, command, command_value = None):
#        if self.active == True:
        if command != Action.NOTHING:
            self.droneController.perform_action(command, command_value)