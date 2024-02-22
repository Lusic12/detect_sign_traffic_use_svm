import numpy as np
import cv2
import sys
import time
np.set_printoptions(threshold=sys.maxsize)
import pickle
from predict import get_blue_countours,get_red_predicted_image,get_red_contours,get_blue_predicted_image


#load model
with open("D:/pythonProject1/red.pkl", "rb") as f:
    red_clf = pickle.load(f)
with open("D:/pythonProject1/blue.pkl", "rb") as f:
    blue_clf = pickle.load(f)

#label
red_labels = ["forbit","stop"]
blue_labels = ["cross way","parking","roundabout","traight"]

#test with image
def detect(image):
    frame = image.copy()
    fin = image.copy()
    img_ihls = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS_FULL)

    red_cnts, red_hulls = get_red_contours(img_ihls)
    #red detect
    if not red_cnts == []:
            red_pred, red_cnt = get_red_predicted_image(red_cnts, frame, red_hulls, red_clf)
            if red_pred is not None:
                for i in range(len(red_cnts)):
                    x,y,w,h = cv2.boundingRect(red_cnt)
                    cv2.rectangle(fin, (x,y), (int(x+w) , int(y+h)) , (0,255,0),2)
                    new_x = x - 64
                    new_y = y + 64
                    label = red_labels[red_pred]
                    #cv2.putText(fin,f"SIZE: {int(w*h)}", (new_x+5, new_y+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.putText(fin, label, (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #blue detect
    blue_cnts, blue_hulls = get_blue_countours(img_ihls)
    if not blue_cnts == []:
        # blue_pred, blue_x, blue_y = get_blue_predicted_image(blue_cnts, frame, blue_hulls, blue_classifier)

        blue_pred, blue_cnt = get_blue_predicted_image(blue_cnts, frame, blue_hulls, blue_clf)


        if blue_pred is not None:
            x,y,w,h = cv2.boundingRect(blue_cnt)
            cv2.rectangle(fin, (x,y), (int(x+w) , int(y+h)) , (0,255,0),2)
            new_x_ = x - 64
            new_y_ = y + 64
            label = blue_labels[blue_pred]
            #cv2.putText(fin, f"SIZE: {int(w*h)}", (new_x_+5, new_y_+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(fin, label, (new_x_, new_y_), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return fin


# Open video file
#cap = cv2.VideoCapture("D:/pythonProject1/bfmc2020_online_2.avi")
cap =cv2.VideoCapture(0)

# Initialize variables
frame_count = 0
start_time = time.time()
fps = 0

while True:
    # Read frame from video
    ret, frame = cap.read()

    if not ret:
        # Check if end of video is reached, if so, exit loop
        break

    # Display frame
    #frame = frame[400:700, 1000:, :]  # Crop frame if needed
    print(frame.shape)
    frame = detect(frame)

    # Increment frame count
    frame_count += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Update FPS value every second
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    # Wait for and check keyboard input
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        # If 'q' is pressed, exit loop
        break

# Release video and close display window
cap.release()
cv2.destroyAllWindows()