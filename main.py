from threading import Thread

# Altered bebop controlling code
from bebop_edited import Bebop
import drone_vision_gui_edited as drone_vision_gui
  
# Own code
from drone_controllers import DroneController
from stream_outputs import StreamOutput
from stream_inputs import StreamInput
from keyboard_inputs import KeyboardInput
from offline import OfflineDroneController, OfflineBebop, OfflineDroneVisionGUI


# If online, it will load the bebop stuff while offline will load the webcam stuff.
is_online = False
        
def online():
    bebop = Bebop()
    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        print('success')
        # Initialize the video part
        bebopVision = drone_vision_gui.DroneVisionGUI(bebop, is_bebop=True, user_args=(bebop, ))

        # Initialize the controller
        droneController = DroneController(bebop, is_online)
        streamOutput = StreamOutput()
        
        stream_input = StreamInput(bebopVision, streamOutput, droneController)
        bebopVision.set_user_callback_function(stream_input.processing_stream, user_callback_args=None)
        
        # Initialize the keyboard input part
        keyboardInput = KeyboardInput(droneController)
        thread_KeyboardInput = Thread(target = keyboardInput.wait_for_input, args = ( ))
        thread_KeyboardInput.start()
        
        # Start the video stream
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")


    bebop.disconnect()
    print('Disconnected. Finished.')
    
def offline():
    bebop = OfflineBebop()

    # Initialize the video part
    bebopVision = OfflineDroneVisionGUI(bebop, is_bebop=True, user_args=(bebop, ))

    # Initialize the 'fake' controller
    droneController = OfflineDroneController(bebop)
    streamOutput = StreamOutput()
    
    # Initialize the keyboard input part
    keyboardInput = KeyboardInput(droneController)
    thread_KeyboardInput = Thread(target = keyboardInput.wait_for_input, args = ( ))
    thread_KeyboardInput.start()
    
    # Act as the open_video function which basically will update the stream.
    stream_input = StreamInput(bebopVision, streamOutput, droneController)
    while True:
        stream_input.processing_stream(None)
    
if __name__ == "__main__":
    if is_online:
        online()
    else:
        offline()