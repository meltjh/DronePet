from enum import Enum

class Action(Enum):
    NOTHING = -1
    ABORT = 0 # TODO
    LOOK_LEFT = 1
    LOOK_RIGHT = 2
    LOOK_UP = 3
    LOOK_DOWN = 4
    LOOK_CENTER = 5 # TODO