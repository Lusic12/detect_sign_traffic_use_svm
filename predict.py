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

# Hàm tiền xử lý ảnh cho vùng màu đỏ
def preprocess_red(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)
    resized_img = resize(img_gray, output_shape=(64, 64), anti_aliasing=True)
    hog_feature = feature.hog(resized_img, orientations=4, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1", visualize=False,
                              feature_vector=True)
    return hog_feature

# Hàm tiền xử lý ảnh cho vùng màu xanh
def preprocess_blue(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)
    resized_img = resize(img_gray, output_shape=(64, 64), anti_aliasing=True)
    hog_feature = feature.hog(resized_img, orientations=4, pixels_per_cell=(4, 4),
                              cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1", visualize=False,
                              feature_vector=True)
    return hog_feature

# Hàm tiền xử lý ảnh cho vùng màu xanh lá cây


# Hàm dự đoán vùng màu đỏ trong ảnh
def predict_img_red(input_img, classifier):
    normalized_img = preprocess_red(input_img)
    y_pred = classifier.predict([normalized_img])[0]
    y_pred_prob = classifier.predict_proba([normalized_img])[0]
    return y_pred, y_pred_prob

# Hàm dự đoán vùng màu xanh trong ảnh
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
    # chỉnh lại lower và upper cho hợp lí (mở rộng ra)
    lower_blue_1 = np.array([94, 115, 0])
    upper_blue_1 = np.array([126, 255, 200])
    mask = cv2.inRange(image_hsv, lower_blue_1, upper_blue_1)


    # mask
    blue_mask_ = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    l_channel = blue_mask_[:, :, 2]
    s_channel = blue_mask_[:, :, 1]
    h_channel = blue_mask_[:, :, 0]

    filtered_r = cv2.medianBlur(l_channel, 5)
    filtered_g = cv2.medianBlur(s_channel, 3)
    filtered_b = cv2.medianBlur(h_channel, 3)
    # filter màu xanh
    filtered_red = -0 * filtered_r + 10 * filtered_b - 0 * filtered_g
    regions, _ = mser_blue.detectRegions(np.uint8(filtered_red))

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # khoanh vùng
    blank = np.zeros_like(blue_mask_)

    cv2.fillPoly(np.uint8(blank), hulls, (0, 0, 255))

    kernel_2 = np.ones((1, 1), np.uint8)
    # mở rộng vùng
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel_2)

    resized_eroded = cv2.resize(opening, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    _, r_thresh = cv2.threshold(opening[:, :, 2], 30, 255, cv2.THRESH_BINARY)

    small_blank = np.zeros((64, 64))

    cnts,_ = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
    return cnts, hulls


def get_red_contours(img_ihls):
    # Sao chép ảnh đầu vào để không làm thay đổi ảnh gốc
    img_hls = img_ihls.copy()

    # Cân bằng histogram của kênh sáng (L) trong không gian màu HLS
    img_hls[:, :, 1] = cv2.equalizeHist(img_hls[:, :, 1])

    # Chuyển đổi không gian màu từ HLS sang BGR
    img_bgr = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR_FULL)

    # Chuyển đổi ảnh BGR sang HSV để phân đoạn vùng màu đỏ
    image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Đặt ngưỡng cho phân đoạn vùng màu đỏ
    lower_red_1 = np.array([130, 60, 2])
    upper_red_1 = np.array([180, 255, 255])
    mask_1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)

    # Áp dụng mask để chỉ giữ lại các pixel thuộc vùng màu đỏ
    red_mask_ = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_1)

    # Tách các kênh màu
    l_channel = red_mask_[:, :, 2]
    s_channel = red_mask_[:, :, 1]
    h_channel = red_mask_[:, :, 0]

    # Lọc ảnh kênh màu đỏ bằng median blur
    filtered_r = cv2.medianBlur(l_channel, 5)
    filtered_g = cv2.medianBlur(s_channel, 5)
    filtered_b = cv2.medianBlur(h_channel, 5)

    # Tính toán ảnh kênh màu đỏ lọc
    filtered_red = 10 * filtered_r - 0 * filtered_b + 0 * filtered_g

    # Phát hiện các vùng MSER trong ảnh kênh màu đỏ lọc
    regions, _ = mser_red.detectRegions(np.uint8(filtered_red))

    # Lấy các đa giác lồi bao quanh các vùng MSER
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # Tạo một ảnh trắng có cùng kích thước với ảnh màu đỏ
    blank = np.zeros_like(red_mask_)

    # Tô màu các đa giác lồi lên ảnh trắng
    cv2.fillPoly(np.uint8(blank), hulls, (0, 0, 255))

    # Mở rộng các vùng màu đỏ bằng phép mở rộng morpohology
    kernel_2 = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel_2)

    # Phóng to ảnh sau khi mở rộng
    resized_eroded = cv2.resize(opening, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # Áp dụng ngưỡng cho kênh màu đỏ sau khi mở rộng
    _, r_thresh = cv2.threshold(opening[:, :, 2], 60, 255, cv2.THRESH_BINARY)
    cv2.imshow('red',r_thresh)
    # Tìm các đường viền trong ảnh đã được ngưỡng
    cnts, _ = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts, hulls


def get_red_predicted_image(cnts, frame, hulls, red_clf):
    max_cnts = 3
    pred_im = None
    x_ = None
    y_ = None

    prediction_list = []
    countor_list = []
    proba_list = []

    # Sắp xếp các đường viền theo diện tích giảm dần
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Giới hạn số lượng đường viền lớn nhất
    if len(cnts_sorted) > max_cnts:
        cnts_sorted = cnts_sorted[:3]

    # Duyệt qua các đường viền đã sắp xếp
    for c in cnts_sorted:
        x_, y_, w, h = cv2.boundingRect(c)
        # Kiểm tra điều kiện về kích thước và tỉ lệ khung hình
        if x_ < 100 or h < 50 or w * h < 8000  or w*h > 20000 or w / h <= 0.9 or w / h > 1.1:
            continue
        # Tạo mask trắng có kích thước giống với frame
        mask = np.zeros_like(frame)

        # Vẽ đường viền lên mask
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # Cắt tấm ảnh theo kích thước của bounding box
        out = frame[y_:y_ + h, x_:x_ + w]

        # Dự đoán ảnh cắt được
        pred, pred_proba = predict_img_blue(out, red_clf)

        # Lưu dự đoán và các thông tin liên quan
        proba_list.append(max(pred_proba))
        prediction_list.append(pred)
        countor_list.append(c)

    # Lựa chọn dự đoán có xác suất lớn nhất
    predict = None
    countor = []
    if not proba_list == [] and max(proba_list) > 0.5:
        _max = max(proba_list)
        for i in range(len(proba_list)):
            if proba_list[i] == _max:
                predict = prediction_list[i]
                countor = countor_list[i]

    return predict, countor

def get_blue_predicted_image(cnts, frame, hulls, blue_clf):
    max_cnts = 3
    # Khởi tạo các biến
    pred_im = None
    x_ = None
    y_ = None

    # Danh sách lưu trữ dự đoán, đường viền và xác suất
    prediction_list = []
    contour_list = []
    probability_list = []

    # Sắp xếp danh sách đường viền theo diện tích giảm dần
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Giới hạn số lượng đường viền lớn nhất
    if len(cnts_sorted) > max_cnts:
        cnts_sorted = cnts_sorted[:3]

    # Duyệt qua các đường viền đã sắp xếp
    for c in cnts_sorted:
        # Lấy thông tin về hình chữ nhật bao quanh đường viền
        x_, y_, w, h = cv2.boundingRect(c)

        # Kiểm tra các điều kiện về kích thước và tỉ lệ khung hình
        if x_ < 100 or h < 100 or w * h < 8000 or w * h > 17000 or w / h <= 0.9 or w / h > 1.1:
            continue

        # Sao chép frame để không làm thay đổi frame gốc
        new_frame = np.copy(frame)

        # Vẽ đường viền lên frame
        cv2.drawContours(frame, hulls, -1, (0, 255, 0), 2)

        # Tạo mask để chỉ giữ lại phần nằm trong đường viền
        mask = np.zeros_like(frame)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # Tạo mask trắng có kích thước giống với frame
        mask = np.zeros_like(frame)

        # Vẽ đường viền lên mask
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

        # Tìm bounding box của contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Cắt tấm ảnh theo kích thước của bounding box
        out = new_frame[y:y + h, x:x + w]

        # Dự đoán ảnh cắt được
        pred, pred_proba = predict_img_blue(out, blue_clf)

        # Lưu dự đoán và các thông tin liên quan
        probability_list.append(max(pred_proba))
        prediction_list.append(pred)
        contour_list.append(c)

    # Lựa chọn dự đoán có xác suất lớn nhất
    predict = None
    contour = []

    if not probability_list == [] and max(probability_list) > 0.5:
        _max = max(probability_list)
        for i in range(len(probability_list)):
            if probability_list[i] == _max:
                predict = prediction_list[i]
                contour = contour_list[i]

    return predict, contour
