import cv2
import numpy as np
print(path)
train_benign = os.path.join(path, f"train/benign")
train_malignant = os.path.join(path, f"train/malignant")
test_benign = os.path.join(path, f"test/benign")
test_malignant = os.path.join(path, f"test/malignant")
def convert_img_to_MC1():
    for i in os.listdir(train_benign):
        image = cv2.imread(os.path.join(train_benign,i))
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for a specific color (e.g., green)
        # These values will vary depending on the exact shade of the color and lighting conditions
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        # Create a mask for the green color
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        print(result)

convert_img_to_MC1()
def convert_img_to_multi_color_IBC():
    ''' 
        Return: list of list of features, [MC1, MC2, MC3, MC4]
    '''


