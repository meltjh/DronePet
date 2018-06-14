from enum import Enum

class Action(Enum):
    NOTHING = -1
    ABORT = 0 # TODO
    
    # Camera
    LOOK_LEFT = 1
    LOOK_RIGHT = 2
    LOOK_UP = 3
    LOOK_DOWN = 4
    
    #
    LOOK_CENTER = 5 # TODO
    TEST = 6 # To test something while debugging
    TAKEOFF = 7
    SAVELAND = 8
    
    # Move
    MOVE_FORWARD = 9
    MOVE_BACKWARD = 10
    MOVE_LEFT = 11
    MOVE_RIGHT = 12
    MOVE_UP = 13
    MOVE_DOWN = 14
    ROTATE_LEFT = 15
    ROTATE_RIGHT = 16