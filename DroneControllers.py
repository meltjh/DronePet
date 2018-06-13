from Actions import Action
import time

class DroneController:
    communication = None
    bebop = None
    
    def __init__(self, communication, bebop):
        print('Controller')
        self.communication = communication
        self.bebop = bebop
     
    def listen_to_controls(self):
        print('listen_to_controls')
        while self.communication.active == True:
            time.sleep(0.05)
            
            if self.communication.last_command != -1:
                print(self.communication.last_command)
                
                self.command_to_action(self.communication.last_command)

                self.communication.last_command = -1
       
    def command_to_action(self, cmd):
        if cmd == Action.LOOK_UP:
            self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=+16, duration=1)
            return
        if cmd == Action.LOOK_DOWN:
            self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-16, duration=1)
            return
        if cmd == Action.LOOK_LEFT:
            self.bebop.pan_tilt_camera_velocity(pan_velocity=-16, tilt_velocity=0, duration=1)
            return
        if cmd == Action.LOOK_RIGHT:
            self.bebop.pan_tilt_camera_velocity(pan_velocity=+16, tilt_velocity=0, duration=1)
            return