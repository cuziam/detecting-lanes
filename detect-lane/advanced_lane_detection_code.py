
import cv2
import numpy as np
import matplotlib.pyplot as plt

def advanced_lane_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply Canny edge detection
    edges_image = cv2.Canny(adaptive_threshold, 50, 150)
    
    # Define a region of interest (ROI)
    height, width = edges_image.shape
    roi_vertices = [
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height),
    ]
    roi_vertices = np.array([roi_vertices], dtype=np.int32)
    
    # Create a mask with the ROI
    mask = np.zeros_like(edges_image)
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Apply the mask to the edges image
    roi_image = cv2.bitwise_and(edges_image, mask)
    
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Create an image to draw the lines
    lines_image = np.zeros_like(image)
    
    # Draw the detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Combine the original image with the lines image
    combined_image = cv2.addWeighted(image, 0.8, lines_image, 1, 0)
    
    return combined_image
