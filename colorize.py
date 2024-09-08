# Import statements
import numpy as np
import argparse
import cv2
import os

"""
Download the model files: 
	1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models 
	2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
	3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1 
 
	Download the no.3 module as no.1 already existed hai
"""

# Paths to load the model
# New directory path
DIR = r"C:\Users\HP\Desktop\IMAGE COLOURIZER"

# Output folder for colorized images
OUTPUT_DIR = r"C:\Users\HP\Desktop\IMAGE COLOURIZER\output"

# Updated paths to model files
PROTOTXT = os.path.join(DIR, r"model\colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model\pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model\colorization_release_v2.caffemodel")


# Argparser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input black and white image")
args = vars(ap.parse_args())

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

# Show the original and colorized images for preview
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)

# Wait for a key press before continuing (this allows user to preview first)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Close the windows after preview

# Function to generate unique filenames with counter
def get_unique_filename(directory, base_name, extension):
    counter = 1
    file_path = os.path.join(directory, f"{base_name}{counter}{extension}")
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(directory, f"{base_name}{counter}{extension}")
    return file_path

# After preview, ask if the user wants to save the colorized image
save_image = input("Do you want to save the colorized image? (yes/no): ").strip().lower()

if save_image == "yes":
    # Get a unique filename in the output folder
    output_path = get_unique_filename(OUTPUT_DIR, "colorized_image_", ".jpg")
    cv2.imwrite(output_path, colorized)
    print(f"Colorized image saved at {output_path}")
else:
    print("Image not saved.")

# python colorize.py -i images\nature.jpg
