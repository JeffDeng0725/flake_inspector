import cv2
import numpy as np
import os

def process_images_in_folder(input_folder, output_folder, threshold_range, n, channel):
    low_threshold, high_threshold = threshold_range
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)
        
        if channel == 'red':
            channel_image = image[:, :, 2]
        elif channel == 'blue':
            channel_image = image[:, :, 0]
        elif channel == 'green':
            channel_image = image[:, :, 1]
        elif channel == 'gray':
            channel_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Example processing (this should be replaced with your actual processing logic)
        for i in range(n):
            threshold_value = low_threshold + (high_threshold - low_threshold) * i / (n - 1)
            _, threshold_image = cv2.threshold(channel_image, threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_threshold_{threshold_value}.jpg")
            cv2.imwrite(output_path, contour_image)

