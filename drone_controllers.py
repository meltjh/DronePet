from actions import Action

class DroneController:
    bebop = None
    
    def __init__(self, bebop, is_online):
        print('Controller')
        self.bebop = bebop
        self.bebop.SetMaxRotationSpeed(35)
        # To ensure a certain quality of the stream if not set already.
        # self.bebop.video_resolution_mode("rec720_stream720")
        self.is_online = is_online
        
        # To allow flip, this should be set during runtime for savety.
        self.allow_flip = False

    def perform_action(self, cmd, value=None, duration=1):

        if cmd == Action.NOTHING:
            return
        
        print(cmd)
        
        # Camera control      
        if cmd == Action.LOOK_UP:
            if value is None:
                self.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=+16, duration=duration)
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

        # Moving control
        if self.is_online:
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
                self.bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=25, duration=duration)
                return
            if cmd == Action.MOVE_DOWN:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-25, duration=duration)
                return
            if cmd == Action.ROTATE_LEFT:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=-50, vertical_movement=0, duration=duration)
                return
            if cmd == Action.ROTATE_RIGHT:
                self.bebop.fly_direct(roll=0, pitch=0, yaw=50, vertical_movement=0, duration=duration)
                return
                        
            # Take off and landing control
            if cmd == Action.ABORT:
                self.bebop.emergency_land()
                return
            if cmd == Action.TAKEOFF:
                self.bebop.safe_takeoff(2)
                return
            if cmd == Action.SAVELAND:
                self.bebop.safe_land(5)
                return

            # Controls for flipping
            if cmd == Action.ALLOW_FLIP:
                self.allow_flip = True
                return
            if cmd == Action.DISALLOW_FLIP:
                self.allow_flip = False
                return
            if cmd == Action.FLIP:
                if self.allow_flip:
                    self.bebop.flip("back")            
                else:
                    print("Action refused: flipping is not allowed, ALLOW_FLIP first!")
                return
        