from Bebop_edited import Bebop
import DroneVisionGUI_edited as DroneVisionGUI



from threading import Thread


from Communications import Communication   
from KeyboardInputs import KeyboardInput
from WebsiteOutputs import WebsiteOutput
from DroneControllers import DroneController
from StreamInputs import StreamInput

        

        
if __name__ == "__main__":
    bebop = Bebop()
    

    # connect to the bebop
    success = bebop.connect(3)

    if (success):
        print('success')
        # start up the video
        bebopVision = DroneVisionGUI.DroneVisionGUI(bebop, is_bebop=True, user_args=(bebop, ))



        droneController = DroneController(bebop)
        communication = Communication(droneController)
        

        stream_input = StreamInput(communication, bebopVision)
        bebopVision.set_user_callback_function(stream_input.processing_stream, user_callback_args=None)
        
        
        keyboardInput = KeyboardInput(communication)
        thread_KeyboardInput = Thread(target = keyboardInput.wait_for_input, args = ( ))
        thread_KeyboardInput.start()
        
        
        websiteOutput = WebsiteOutput(communication)
        thread_WebsiteOutput = Thread(target = websiteOutput.show_output, args = ( ))
        thread_WebsiteOutput.start()
        
        
#        droneController = DroneController(communication, bebop)
#        thread_Controller = Thread(target = droneController.listen_to_controls, args = ( ))
#        thread_Controller.start()
        
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")


    bebop.disconnect()
    print('Bye bye')