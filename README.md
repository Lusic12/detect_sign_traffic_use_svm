# Biển Báo Recognition using Support Vector Machine

This project aims to detect traffic signs using Support Vector Machine (SVM) and color space filtering. It is designed to run on Raspberry Pi 4B with a frame size of 320x240 and a frame rate of 10 frames per second.

## Installation

1. Clone the repository:


2. Install required dependencies:
    ```bash
    pip install package from requirements.txt
    ```

## Usage

1. Connect your Raspberry Pi camera module.
   ```
    from picamera2 import Picamera2
    piCam =Picamera2()
    piCam.preview_configuration.main.size=(1280,720)
    piCam.preview_configuration.main.format="RGB888"
    piCam.preview_configuration.align()
    piCam.configure("preview")
    piCam.start()
    while True:
        frame=piCam.capture_array()
        ....
   ```

2. Run the main script:
    ```bash
    python main.py
    ```

## Approach

1. **Color Space Filtering**: The input frames are filtered in a specific color space to isolate regions containing traffic signs based on their color characteristics.

2. **Contour Detection**: Contours are detected from the filtered frames, identifying potential regions of interest.

3. **Feature Extraction**: Features are extracted from the detected regions to represent the characteristics of traffic signs.

4. **Classification with SVM**: Support Vector Machine classifier is used to predict the type of traffic signs based on the extracted features.

