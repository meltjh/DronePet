from actions import Action

class KeyboardInput:
    communication = None
    
    def __init__(self, droneController):
        self.droneController = droneController
   
    # Infinite loop of asking for an input. The action will be executed directly after hitting enter.
    def wait_for_input(self):
        while True:
            inpt = input("Keyboard input: ")
            self.droneController.perform_action(self.input_to_command(inpt))
    
    # Returns the Action given the textual input.    
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
            'm': Action.ALLOW_FLIP,
            '`': Action.DISALLOW_FLIP,
            'flip': Action.FLIP
        }.get(inpt.lower(), Action.NOTHING) # Default.