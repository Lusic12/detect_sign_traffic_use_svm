import numpy as np
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)
from skimage.transform import resize
from skimage import feature

# Khởi tạo bộ phát hiện MSER cho các màu cụ thể
mser_red = cv2.MSER_create(8, 400, 2000)
mser_blue = cv2.MSER_create(8, 400, 2000)
mser_green = cv2.MSER_create(8, 400, 2000)
mser_yellow = cv2.MSER_create(8, 400, 2000)

# preprocessing_red
def preprocess_red(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)
    resized_img = resize(img_gray, output_shape=(64, 64), anti_aliasing=True)
    hog_feature = feature.hog(resized_img, orientations=4, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1", visualize=False,
                              feature_vector=True)
    return hog_feature

# preprocessing_bule
def preprocess_blue(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)
    resized_img = resize(img_gray, output_shape=(64, 64), anti_aliasing=True)
    hog_feature = feature.hog(resized_img, orientations=4, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1", visualize=False,
                              feature_vector=True)
    return hog_feature



# predict_red
def predict_img_red(input_img, classifier):
    normalized_img = preprocess_red(input_img)
    y_pred = classifier.predict([normalized_img])[0]
    y_pred_prob = classifier.predict_proba([normalized_img])[0]
    return y_pred, y_pred_prob

# predict_blue
def predict_img_blue(input_img, classifier):
    normalized_img = preprocess_blue(input_img)
    y_pred = classifier.predict([normalized_img])[0]
    y_pred_prob = classifier.predict_proba([normalized_img])[0]
    return y_pred, y_pred_prob


def get_blue_countours(img_ihls):
    img_hls = img_ihls.copy()
    img_hls[:, :, 1] = cv2.equalizeHist(img_hls[:, :, 1])
    img_bgr = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR_FULL)

    image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue_1 = np.array([94, 115, 0])
    upper_blue_1 = np.array([126, 255, 200])
    mask = cv2.inRange(image_hsv, lower_blue_1, upper_blue_1)


    # mask_blue
    blue_mask_ = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    l_channel = blue_mask_[:, :, 2]
    s_channel = blue_mask_[:, :, 1]
    h_channel = blue_mask_[:, :, 0]
    # filter blue
    filtered_r = cv2.medianBlur(l_channel, 5)
    filtered_g = cv2.medianBlur(s_channel, 3)
    filtered_b = cv2.medianBlur(h_channel, 3)
    filtered_blue = -0 * filtered_r + 10 * filtered_b - 0 * filtered_g
    regions, _ = mser_blue.detectRegions(np.uint8(filtered_blue))

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # choose area
    blank = np.zeros_like(blue_mask_)

    cv2.fillPoly(np.uint8(blank), hulls, (0, 0, 255))

    kernel_2 = np.ones((1, 1), np.uint8)
    # opeing area
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel_2)

    _, r_thresh = cv2.threshold(opening[:, :, 2], 30, 255, cv2.THRESH_BINARY)

    cnts,_ = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts, hulls


def get_red_contours(img_ihls):
    img_hls = img_ihls.copy()
    img_hls[:, :, 1] = cv2.equalizeHist(img_hls[:, :, 1])
    img_bgr = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR_FULL)
    image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([130, 60, 2])
    upper_red_1 = np.array([180, 255, 255])
    mask_1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)

    # mask_red
    red_mask_ = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_1)

    l_channel = red_mask_[:, :, 2]
    s_channel = red_mask_[:, :, 1]
    h_channel = red_mask_[:, :, 0]

    # filter red

    filtered_r = cv2.medianBlur(l_channel, 5)
    filtered_g = cv2.medianBlur(s_channel, 5)
    filtered_b = cv2.medianBlur(h_channel, 5)

    filtered_red = 10 * filtered_r - 0 * filtered_b + 0 * filtered_g

    regions, _ = mser_red.detectRegions(np.uint8(filtered_red))
    #choose area
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank = np.zeros_like(red_mask_)

    cv2.fillPoly(np.uint8(blank), hulls, (0, 0, 255))

    # openning
    kernel_2 = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel_2)

    resized_eroded = cv2.resize(opening, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    _, r_thresh = cv2.threshold(opening[:, :, 2], 60, 255, cv2.THRESH_BINARY)
    cv2.imshow('red',r_thresh)
    cnts, _ = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts, hulls


def get_red_predicted_image(cnts, frame, hulls, red_clf):
    max_cnts = 3
    pred_im = None
    x_ = None
    y_ = None

    prediction_list = []
    contour_list = []
    proba_list = []

    # Sort contours
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Limit number of contours
    if len(cnts_sorted) > max_cnts:
        cnts_sorted = cnts_sorted[:3]

    for c in cnts_sorted:
        x_, y_, w, h = cv2.boundingRect(c)
        
        # Check conditions regarding contour dimensions
        if x_ < 100 or h < 50 or w * h < 8000 or w * h > 20000 or w / h <= 0.9 or w / h > 1.1:
            continue
        mask = np.zeros_like(frame)

        # Draw contour on mask
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # Crop image
        out = frame[y_:y_ + h, x_:x_ + w]

        # Predict from cropped image
        pred, pred_proba = predict_img_blue(out, red_clf)
        proba_list.append(max(pred_proba))
        prediction_list.append(pred)
        contour_list.append(c)

    predict = None
    contour = []
    if not proba_list == [] and max(proba_list) > 0.5:
        _max = max(proba_list)
        for i in range(len(proba_list)):
            if proba_list[i] == _max:
                predict = prediction_list[i]
                contour = contour_list[i]

    return predict, contour

def get_blue_predicted_image(cnts, frame, hulls, blue_clf):
    max_cnts = 3
    # Initialize variables
    pred_im = None
    x_ = None
    y_ = None

    # Lists to store prediction, contours, and probabilities
    prediction_list = []
    contour_list = []
    probability_list = []

    # Sort contours in descending order of area
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Limit the number of largest contours
    if len(cnts_sorted) > max_cnts:
        cnts_sorted = cnts_sorted[:3]

    # Iterate through sorted contours
    for c in cnts_sorted:
        # Get bounding rectangle information of the contour
        x_, y_, w, h = cv2.boundingRect(c)

        # Check conditions regarding size and aspect ratio of the bounding box
        if x_ < 100 or h < 100 or w * h < 8000 or w * h > 17000 or w / h <= 0.9 or w / h > 1.1:
            continue

        # Copy frame to preserve the original frame
        new_frame = np.copy(frame)

        # Draw contours on the frame
        cv2.drawContours(frame, hulls, -1, (0, 255, 0), 2)

        # Create a mask to keep only the part inside the contour
        mask = np.zeros_like(frame)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # Find bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Crop the image based on the bounding box dimensions
        out = new_frame[y:y + h, x:x + w]

        # Predict the cropped image
        pred, pred_proba = predict_img_blue(out, blue_clf)

        # Store prediction and related information
        probability_list.append(max(pred_proba))
        prediction_list.append(pred)
        contour_list.append(c)

    # Select the prediction with the highest probability
    predict = None
    contour = []

    if not probability_list == [] and max(probability_list) > 0.5:
        _max = max(probability_list)
        for i in range(len(probability_list)):
            if probability_list[i] == _max:
                predict = prediction_list[i]
                contour = contour_list[i]

    return predict, contour
