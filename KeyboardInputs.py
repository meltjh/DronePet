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
            
            self.communication.last_command = self.input_to_command(inpt)
            
    def input_to_command(self, inpt):
        return {
            'w': Action.LOOK_UP,
            's': Action.LOOK_DOWN,
            'a': Action.LOOK_LEFT,
            'd': Action.LOOK_RIGHT
        }.get(inpt, Action.NOTHING) # default 