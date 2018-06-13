from Bebop_edited import Bebop
import DroneVisionGUI_edited as DroneVisionGUI



from threading import Thread


from Communications import Communication   
from GestureInputs import GestureInput
from KeyboardInputs import KeyboardInput
from WebsiteOutputs import WebsiteOutput
from DroneControllers import DroneController

        

        
if __name__ == "__main__":
    bebop = Bebop()
    communication = Communication()

    # connect to the bebop
    success = bebop.connect(3)

    if (success):
        print('success')
        # start up the video
        bebopVision = DroneVisionGUI.DroneVisionGUI(bebop, is_bebop=True, user_args=(bebop, ))


        gesture_input = GestureInput(communication, bebopVision)
        bebopVision.set_user_callback_function(gesture_input.obtain_image, user_callback_args=None)
        
        
        keyboardInput = KeyboardInput(communication)
        thread_KeyboardInput = Thread(target = keyboardInput.wait_for_input, args = ( ))
        thread_KeyboardInput.start()
        
        
        websiteOutput = WebsiteOutput(communication)
        thread_WebsiteOutput = Thread(target = websiteOutput.show_output, args = ( ))
        thread_WebsiteOutput.start()
        
        
        droneController = DroneController(communication, bebop)
        thread_Controller = Thread(target = droneController.listen_to_controls, args = ( ))
        thread_Controller.start()
        
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")


    bebop.disconnect()
    print('Bye bye')