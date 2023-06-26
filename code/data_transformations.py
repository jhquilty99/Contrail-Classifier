import pandas as pd
import shutil
import os 
from PIL import Image

# Separates unlabled roboflow data into contrails/no contrials folders
def Label_Data(CSV, folder):
    # Loads the CSV file
    data = pd.read_csv(CSV, header=None, names=['image', 'x1', 'y1', 'x2', 'y2', 'label'])

    # Get a list of unique image names and a list of all images in the directory
    unique_images = data['image'].unique()
    all_images = os.listdir(folder)

    # Get a list to hold the names of images without contrails
    no_contrail_images = [image for image in all_images if image not in unique_images]


    # Move images with contrails to 'Contrail' directory and move images without contrails to 'No contrail' directory
    for image in unique_images:
         shutil.move(folder + image, '../data/Roboflow/Contrail/')

    for image in no_contrail_images:
        shutil.move(folder + image, '../data/Roboflow/No_contrail/')



CSV = "../data/Roboflow_2/valid/_annotations.csv"
folder = '../data/Roboflow_2/valid/'

Label_Data(CSV, folder)

def image_transformer(image_dir):
    # Iterate over each image in the directory
    for folder in os.listdir(image_dir):
        for filename in os.listdir(os.path.join(image_dir, folder)):
            # Open each image file
            img_path = os.path.join(image_dir, folder, filename)
            img = Image.open(img_path)
            # Convert PNG images to JPEG format
            if filename.endswith(".png"):
                jpeg_path = os.path.splitext(img_path)[0] + ".jpg"
                img = img.convert("RGB")
                img.save(jpeg_path, "JPEG")

image_dir = "../data/Singapore Data Swimcat"
images_singapore, classes_singapore = image_transformer(image_dir)