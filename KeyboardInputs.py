from Actions import Action
import time

class KeyboardInput:
    communication = None
    
    def __init__(self, communication):
        print('KeyboardInput')
        self.communication = communication
   
    def wait_for_input(self):
        print('wait_for_input')
        while self.communication.active == True:
            time.sleep(0.1)
            
            inpt = input("Keyboard input: ")
            
            self.communication.send_command(self.input_to_command(inpt))
            
    def input_to_command(self, inpt):
        return {
            ']': Action.LOOK_UP,
            '\\': Action.LOOK_DOWN,
            '-': Action.LOOK_LEFT,
            '=': Action.LOOK_RIGHT,
            
            'w': Action.MOVE_FORWARD,
            's': Action.MOVE_BACKWARD,
            'a': Action.MOVE_LEFT,
            'd': Action.MOVE_RIGHT,
            
            'o': Action.MOVE_UP,
            'l': Action.MOVE_DOWN,
            'k': Action.ROTATE_LEFT,
            ';': Action.ROTATE_RIGHT,
            
            'z': Action.TAKEOFF,
            'x': Action.SAVELAND,
            
            'q': Action.ABORT,
            't': Action.TEST
        }.get(inpt.lower(), Action.NOTHING) # default 