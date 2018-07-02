from enum import Enum

class Action(Enum):
    NOTHING = -1
    ABORT = 0
    
    # Camera
    LOOK_LEFT = 1
    LOOK_RIGHT = 2
    LOOK_UP = 3
    LOOK_DOWN = 4

    
    # Move
    MOVE_FORWARD = 5
    MOVE_BACKWARD = 6
    MOVE_LEFT = 7
    MOVE_RIGHT = 8
    MOVE_UP = 9
    MOVE_DOWN = 10
    ROTATE_LEFT = 11
    ROTATE_RIGHT = 12
    TAKEOFF = 13
    SAVELAND = 14
    FLIP = 15 # Note that this only works when ALLOW_FLIP is done first.
    
    # Set variables
    ALLOW_FLIP = 16
    DISALLOW_FLIP = 17