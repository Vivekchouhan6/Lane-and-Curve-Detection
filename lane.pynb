import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

def region_of_interest(image1, vert):
    msk = np.zeros_like(image1)
    cv2.fillPoly(msk, vert, 255)
    masked_image = cv2.bitwise_and(image1, msk)
    return masked_image

def draw_lines(image1, lines):
    if lines is None:
        return image1
    
    image1 = np.copy(image1)
    blank_image = np.zeros_like(image1)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return cv2.addWeighted(image1, 0.8, blank_image, 1, 0.0)

def process_image(image2):
    ht, width = image2.shape[:2]
    g_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    b_image = cv2.GaussianBlur(g_image, (5, 5), 0)
    canny_image = cv2.Canny(b_image, 50, 150)
    
    vert = np.array([[(0, ht), (width // 2, ht // 2), (width, ht)]], dtype=np.int32)
    roi_image = region_of_interest(canny_image, vert)
    
    lines = cv2.HoughLinesP(roi_image, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    
    return draw_lines(image2, lines)

def save_frames(video_path, output_folder):
    capture = cv2.VideoCapture(video_path)
    
    if not capture.isOpened():
        print("Error: Could not open video file.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    while True:
        ret, image1 = capture.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, image1)
        frame_count += 1
    
    capture.release()
    print(f"Frames saved in {output_folder}")

def process_frames(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image2 = cv2.imread(image_path)
            processed_image = process_image(image2)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
    
    print(f"Processed frames saved in {output_folder}")

def main():
    video_path = 'path_to_video.mp4'
    frames_folder = 'frames'
    processed_folder = 'processed_frames'
    
    save_frames(video_path, frames_folder)
    process_frames(frames_folder, processed_folder)
    
    for filename in sorted(os.listdir(processed_folder)):
        image_path = os.path.join(processed_folder, filename)
        image2 = cv2.imread(image_path)
        cv2_imshow(image2)
        
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
