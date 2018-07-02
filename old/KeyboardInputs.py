from Actions import Action
import time

class KeyboardInput:
    communication = None
    
    def __init__(self, droneController):
        print('KeyboardInput')
        self.droneController = droneController
   
    def wait_for_input(self):
        print('wait_for_input')
        while True:
            inpt = input("Keyboard input: ")
            self.droneController.perform_action(self.input_to_command(inpt))
            
    def input_to_command(self, inpt):
        return {
            ']': Action.LOOK_UP,
            '\\': Action.LOOK_DOWN,
#            '-': Action.LOOK_LEFT,
#            '=': Action.LOOK_RIGHT,
            
#            'w': Action.MOVE_FORWARD,
#            's': Action.MOVE_BACKWARD,
#            'a': Action.MOVE_LEFT,
#            'd': Action.MOVE_RIGHT,
            
            'a': Action.MOVE_UP,
            's': Action.MOVE_DOWN,
#            'k': Action.ROTATE_LEFT,
#            ';': Action.ROTATE_RIGHT,
            
            'z': Action.TAKEOFF,
            'x': Action.SAVELAND,
            
            'q': Action.ABORT,
            'm': Action.ALLOW_MOVEMENTS,
#            '`': Action.DISALLOW_MOVEMENTS,
#            't': Action.TEST
        }.get(inpt.lower(), Action.NOTHING) # default 