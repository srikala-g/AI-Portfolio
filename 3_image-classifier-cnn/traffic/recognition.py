import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2

MODEL_NAME = 'model/model-50-dc.h5'

def auto_crop_and_preprocess(img_path, target_size=(30, 30)):
    # Load image
    image = cv2.imread(img_path)
    orig = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contour detected. Try a clearer image.")

    # Find the largest contour (assumed to be the sign)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add padding to the bounding box (optional)
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Crop the region of interest
    cropped = orig[y:y+h, x:x+w]

    # Convert to RGB
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # Resize to model's input size
    resized = cv2.resize(cropped_rgb, target_size)

    # Normalize and expand dims for prediction
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, cropped_rgb

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

    # img = image.load_img(img_path, target_size=(30, 30))
    # img_array = image.img_to_array(img)
    # img_array = img_array / 255.0
    # img_array = np.expand_dims(img_array, axis=0)
    # return img_array

def main():
    custom = False
    if len(sys.argv) < 2:
        print("Usage: python recognition.py <traffic_signal_image> <custom>")
        sys.exit(1)
    
    if len(sys.argv) == 3:
        custom = sys.argv[2]


    img_path = sys.argv[1]
    model = load_model(MODEL_NAME)

    print(len(sys.argv))
    print(custom)

    if custom == "True" or custom == "true":
        print("custom auto_crop_and_preprocess")
        img_array, cropped_img = auto_crop_and_preprocess(img_path)
    else:
        print("Using default preprocessing...")
        img_array = preprocess_image(img_path)

    if img_array.shape != (1, 30, 30, 3):
       print(f"Invalid input shape: {img_array.shape}")
       sys.exit(1)


    prediction = model.predict(img_array)

    class_idx = np.argmax(prediction, axis=1)[0]


    print("Softmax output:", prediction)
    print("Predicted class index:", np.argmax(prediction, axis=1)[0])
    print("Confidence:", np.max(prediction))
    confidence = np.max(prediction)

    class_names = [
        "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
        "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)",
        "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", "No vehicles",
        "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
        "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
        "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
        "Beware of ice/snow", "Wild animals crossing", "End of all restrictions", "Turn right ahead",
        "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
        "Keep left", "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
    ]
    print(f"Predicted traffic sign: {class_names[class_idx]}")

    top_3 = prediction[0].argsort()[-3:][::-1]
    for i in top_3:
        print(f"{class_names[i]}: {prediction[0][i]:.4f}")

    model.summary()

    # Optional: display the cropped region
    # plt.imshow(cropped_img)
    # plt.title(f"Predicted class: {class_names[class_idx]}, confidence: {confidence:.2f}")
    # plt.axis("off")
    # plt.show()

    # root = tk.Tk()
    # root.title("Traffic Sign Recognition")

    # img_tk = image.load_img(img_path)
    # img_tk = img_tk.resize((128, 128))
    # img_np = np.array(img_tk)
    # img_disp = image.array_to_img(img_np)
    # img_disp = img_disp.resize((128, 128))
    # img_photo = tk.PhotoImage(file=img_path)

    # label_img = tk.Label(root, image=img_photo)
    # label_img.pack(pady=10)

    # label_pred = tk.Label(root, text=f"Prediction: {class_names[class_idx]}", font=("Arial", 16))
    # label_pred.pack(pady=10)

    # root.mainloop()

if __name__ == "__main__":
    main()
