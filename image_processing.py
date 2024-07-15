import cv2
import numpy as np
import os

def crop_center(image, crop_ratio=0.70):
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    crop_height, crop_width = int(image.shape[0] * crop_ratio / 2), int(image.shape[1] * crop_ratio / 2)
    cropped_image = image[center_y - crop_height:center_y + crop_height, center_x - crop_width:center_x + crop_width]
    return cropped_image

def gaussian_blur_subtract(image, ksize=33):
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    sub_image = cv2.subtract(image, blurred_image)
    return cv2.GaussianBlur(sub_image, (ksize, ksize), 0) + image

def apply_multiple_radial_thresholds(image, threshold_range=[100, 170], n=5):
    t_min, t_max = threshold_range
    thresholds = np.linspace(t_min, t_max, n)
    
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    results = []
    for threshold_value in thresholds:
        radial_threshold = threshold_value + (dist_from_center / max_dist) * (threshold_value - threshold_value)

        thresholded_img = np.zeros_like(image)
        thresholded_img[image > radial_threshold] = 255

        contours, _ = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [cnt for cnt in contours if 10000 < cv2.contourArea(cnt)]

        results.append((thresholded_img, large_contours, threshold_value))
    
    return results

def equalize_brightness(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)
    return equalized_image

def calculate_gradient(image, rect):
    x, y, w, h = rect
    gradients = []
    for i in range(y, min(y + h, image.shape[0])):
        row_gradients = []
        for j in range(x, min(x + w - 1, image.shape[1] - 1)):
            gradient = int(image[i, j + 1]) - int(image[i, j])
            row_gradients.append(gradient)
        gradients.append(row_gradients)
    return gradients

def process_image(image_path, output_dir, threshold_range=[100, 170], n=5, channel='gray'):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image {image_path}")
        return

    if channel == 'red':
        channel_image = image[:, :, 2]
    elif channel == 'blue':
        channel_image = image[:, :, 0]
    elif channel == 'green':
        channel_image = image[:, :, 1]
    elif channel == 'gray':
        red_channel = image[:, :, 2]
        blue_channel = image[:, :, 0]
        channel_image = cv2.addWeighted(red_channel, 0, blue_channel, 1, 0)
    else:
        raise ValueError("Invalid channel selected")

    channel_image_eq = equalize_brightness(channel_image)
    gray_image = channel_image_eq

    gray_image_blur_sub = gaussian_blur_subtract(gray_image)
    gray_image_blur_sub_crop = crop_center(gray_image_blur_sub)

    results = apply_multiple_radial_thresholds(gray_image_blur_sub_crop, threshold_range, n)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    processed_files = []

    for i, (thresholded_img, large_contours, threshold_value) in enumerate(results):
        thresholded_output_path = os.path.join(output_dir, f"{name}_threshold_{int(threshold_value)}{ext}")
        contour_output_path = os.path.join(output_dir, f"{name}_contours_{int(threshold_value)}{ext}")

        cv2.imwrite(thresholded_output_path, thresholded_img)

        # Draw contours in color on the original image and calculate gradients
        contour_img = crop_center(image.copy())
        gradients = []
        for contour in large_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = max(0, x - 10), max(0, y - 10), min(image.shape[1] - x, w + 20), min(image.shape[0] - y, h + 20)  # Add padding and ensure within bounds
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Draw rectangle in purple
            contour_gradient = calculate_gradient(gray_image_blur_sub_crop, (x, y, w, h))
            gradients.append(contour_gradient)

        contour_img = cv2.drawContours(contour_img, large_contours, -1, (0, 255, 0), 2)  # Original contours in green
        cv2.imwrite(contour_output_path, contour_img)

        processed_files.append({
            'original': base_name,
            'threshold': int(threshold_value),
            'threshold_image': os.path.basename(thresholded_output_path),
            'contour_image': os.path.basename(contour_output_path),
            'gradient': gradients  # Store gradient information
        })

    return processed_files
