from Actions import Action
#import time

class DroneController:
    bebop = None
    
    def __init__(self, bebop, ONLINE):
        print('Controller')
        self.bebop = bebop
        self.bebop.SetMaxRotationSpeed(30)
        self.ONLINE = ONLINE
        self.allow_movements = False
       
    def perform_action(self, command, command_value=None, duration=1):
        if command != -1 and command != Action.NOTHING:
            print(command)
            self.command_to_action(command, command_value, duration)
        
    def command_to_action(self, cmd, value=None, duration=1):

        # Camera stuff        
        if cmd == Action.LOOK_UP:
            if value is None:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=+16, duration=duration)
                print("woohoo")
            else:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=+value, duration=duration)
            return
        if cmd == Action.LOOK_DOWN:
            if value is None:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-16, duration=duration)
            else:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-value, duration=duration)
            return
        if cmd == Action.LOOK_LEFT:
            if value is None:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=-16, tilt_velocity=0, duration=duration)
            else:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=-value, tilt_velocity=0, duration=duration)
            return
        if cmd == Action.LOOK_RIGHT:
            if value is None:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=+16, tilt_velocity=0, duration=duration)
            else:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=+value, tilt_velocity=0, duration=duration)
            return


        if cmd == Action.ALLOW_MOVEMENTS:
            print("ALLOW_MOVEMENTS")
            self.allow_movements = True
            return
        if cmd == Action.DISALLOW_MOVEMENTS:
            print("DISALLOW_MOVEMENTS")
            self.allow_movements = False
            return
        
        if self.ONLINE and self.allow_movements:
            # Movement stuff
            if cmd == Action.MOVE_FORWARD:
                self.bebop.fly_direct(roll=0, pitch=+50, yaw=0, vertical_movement=0, duration=duration)
                return
            if cmd == Action.MOVE_BACKWARD:
                self.bebop.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=duration)
                return
            if cmd == Action.MOVE_LEFT:
                self.bebop.fly_direct(roll=-50, pitch=0, yaw=0, vertical_movement=0, duration=duration)
                return
            if cmd == Action.MOVE_RIGHT:
                self.bebop.fly_direct(roll=+50, pitch=0, yaw=0, vertical_movement=0, duration=duration)
                return
            if cmd == Action.MOVE_UP:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=50, duration=duration)
                return
            if cmd == Action.MOVE_DOWN:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-50, duration=duration)
                return
            if cmd == Action.ROTATE_LEFT:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=-100, vertical_movement=0, duration=duration)
                return
            if cmd == Action.ROTATE_RIGHT:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=100, vertical_movement=0, duration=duration)
                return
            
            
            # Test stuff
            if cmd == Action.ABORT:
                print('ABORT')
                self.bebop.emergency_land()
                return
            if cmd == Action.TAKEOFF:
                print('TAKEOFF')
                self.bebop.safe_takeoff(2)
                return
            if cmd == Action.SAVELAND:
                print('SAVELAND')
    #            self.bebop.land()
                self.bebop.safe_land(5)
                return
            
    #        if cmd == Action.TEST:
    #            self.bebop.reset()
    #            return
            
    #        if cmd == Action.TEST:
    #            self.bebop.video_stabalisation_mode(mode="none")
    #            time.sleep(5)
    #            self.bebop.video_stabalisation_mode(mode="roll_pitch")            
    #            return
                
            if cmd == Action.TEST:
                self.bebop.flip("back")
                
                
    #            self.bebop.SetMotionDetection(False)
    #            time.sleep(5)
    #            self.bebop.SetMotionDetection(True)
                
                
                
    #            self.bebop.video_resolution_mode("rec720_stream720")
    #            time.sleep(5)
    #            self.bebop.video_resolution_mode("rec1080_stream480")            
                return
        