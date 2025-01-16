import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import av  # Import the av module for video frame handling

class Button:
    def __init__(self, pos, text, size=[100, 100]):
        self.pos = pos
        self.size = size
        self.text = text

# Function to process the video frame from the webcam
def process_video_frame(frame, detector, segmentor, imgList, indexImg, keys, session_state):
    # Convert the frame to a numpy array (BGR format)
    image = frame.to_ndarray(format="bgr24")
    
    # Remove background using SelfiSegmentation
    imgOut = segmentor.removeBG(image, imgList[indexImg])

    # Detect hands on the background-removed image
    hands, img = detector.findHands(imgOut, flipType=False)
    
    # Create a blank canvas for the keyboard
    keyboard_canvas = np.zeros_like(img)
    buttonList = []

    # Create buttons for the virtual keyboard based on the keys list
    for key in keys[0]:
        buttonList.append(Button([30 + keys[0].index(key) * 105, 30], key))
    for key in keys[1]:
        buttonList.append(Button([30 + keys[1].index(key) * 105, 150], key))
    for key in keys[2]:
        buttonList.append(Button([30 + keys[2].index(key) * 105, 260], key))

    # Draw the buttons on the keyboard canvas
    for button in buttonList:
        x, y = button.pos
        cv2.rectangle(keyboard_canvas, (x, y), (x + button.size[0], y + button.size[1]), (255, 255, 255), -1)
        cv2.putText(keyboard_canvas, button.text, (x + 20, y + 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 3)

    # Handle input and gestures from detected hands
    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            if lmList:
                # Get the coordinates of the index finger tip (landmark 8)
                x8, y8 = lmList[8][0], lmList[8][1]
                for button in buttonList:
                    bx, by = button.pos
                    bw, bh = button.size
                    # Check if the index finger is over a button
                    if bx < x8 < bx + bw and by < y8 < by + bh:
                        # Highlight the button and update the text
                        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), -1)
                        cv2.putText(img, button.text, (bx + 20, by + 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)
                        
                        # Update the output text in session_state
                        session_state["output_text"] += button.text

    # Corrected return: Create a video frame from the ndarray image
    return av.VideoFrame.from_ndarray(img, format="bgr24")  # Corrected this line