from Bebop_edited import Bebop
import DroneVisionGUI_edited as DroneVisionGUI



from threading import Thread

  
from KeyboardInputs import KeyboardInput
from StreamOutputs import StreamOutput
from DroneControllers import DroneController
from StreamInputs import StreamInput

from Offline import OfflineDroneController, OfflineBebop, OfflineDroneVisionGUI


ONLINE = False
        
def online():
    bebop = Bebop()
    

    # connect to the bebop
    success = bebop.connect(5)

    if (success):
        print('success')
        # start up the video
        bebopVision = DroneVisionGUI.DroneVisionGUI(bebop, is_bebop=True, user_args=(bebop, ))



        droneController = DroneController(bebop)
        streamOutput = StreamOutput()

        stream_input = StreamInput(bebopVision, streamOutput, droneController)
        bebopVision.set_user_callback_function(stream_input.processing_stream, user_callback_args=None)
        
        
        keyboardInput = KeyboardInput(droneController)
        thread_KeyboardInput = Thread(target = keyboardInput.wait_for_input, args = ( ))
        thread_KeyboardInput.start()
        
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")


    bebop.disconnect()
    print('Bye bye')
    
def offline():
    bebop = OfflineBebop()


    print('success')
    # start up the video
    bebopVision = OfflineDroneVisionGUI(bebop, is_bebop=True, user_args=(bebop, ))



    droneController = OfflineDroneController(bebop)
    streamOutput = StreamOutput()
    
    keyboardInput = KeyboardInput(droneController)
    thread_KeyboardInput = Thread(target = keyboardInput.wait_for_input, args = ( ))
    thread_KeyboardInput.start()
    
    stream_input = StreamInput(bebopVision, streamOutput, droneController)
    while True:
        stream_input.processing_stream(None)
    
if __name__ == "__main__":
    if ONLINE:
        online()
    else:
        offline()