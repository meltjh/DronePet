"""
Demo of the Bebop vision using DroneVisionGUI (relies on libVLC).  It is a different
multi-threaded approach than DroneVision
Author: Amy McGovern
"""
from Bebop_edited import Bebop
import DroneVisionGUI_edited as DroneVisionGUI
import threading
import cv2
import time
#import tf-openpose.run_TEST as tf_pose

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tf_pose
import matplotlib.pyplot as plt

import numpy as np

isAlive = False

class UserVision:
    def __init__(self, vision):
        self.vision = vision
        
        self.h = 432
        self.w = 368

        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(self.h, self.w))
        
    def show_detection(self, args):
        
        img, humans, image_with_human = self.vision.get_latest_valid_picture()
        
        if (image_with_human is not None):
            
            print('\007') # sound
            print('humans',humans)
            print('len',len(humans))
            
            plt.imshow(image_with_human)
            plt.show()
        else:
            print('Geen humans')
            
def poepje(a):
    print(a)

def demo_user_code_after_vision_opened(bebopVision, args):
    bebop = args[0]

    print("Vision successfully started!")
    #removed the user call to this function (it now happens in open_video())
    #bebopVision.start_video_buffering()

    # takeoff
    #bebop.safe_takeoff(5)

    # skipping actually flying for safety purposes indoors - if you want
    # different pictures, move the bebop around by hand
    print("Fly me around by hand!")
    bebop.smart_sleep(0.5)
    
    if (bebopVision.vision_running):
        print("Moving the camera using velocity")
        bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=+32, duration=1)
        bebop.smart_sleep(10)

        # land
        #bebop.safe_land(5)

        print("Finishing demo and stopping vision")
        bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()

if __name__ == "__main__":
    # make my bebop object
    bebop = Bebop()

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        # start up the video
        bebopVision = DroneVisionGUI.DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=demo_user_code_after_vision_opened,
                                     user_args=(bebop, ))

        userVision = UserVision(bebopVision)
        
        bebopVision.set_user_callback_function(poepje, user_callback_args="hoi")
        bebopVision.set_user_callback_function(userVision.show_detection, user_callback_args=None)
        
        bebopVision.open_video()

    else:
        print("Error connecting to bebop.  Retry")

    print('done')