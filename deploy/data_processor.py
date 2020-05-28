import argparse
import os
import cv2
from keras.models import load_model
import pandas as pd
from utils import data_processor_configs
from utils.mask_classifier import classify_image_mask_usage

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--date", required=True,
	help="date of folder which contains images to process")
ap.add_argument("-c", "--mask_data_csv", default='mask_data.csv',
	help="name (or path and name) of csv which contains mask data. Include csv extension in argument")
args = vars(ap.parse_args())

print("[INFO] loading object detection model...")
person_detection_net = cv2.dnn.readNetFromCaffe(data_processor_configs.person_detect_prototxt, data_processor_configs.person_detect_model)

print("[INFO] loading face detection model...")
face_detection_net = cv2.dnn.readNetFromCaffe(data_processor_configs.face_detect_prototxt, data_processor_configs.face_detect_model)

print("[INFO] loading mask detection model...")
mask_model = load_model(data_processor_configs.mask_no_mask_model)

current_wd = os.getcwd()
current_img_folder = current_wd + '/' + data_processor_configs.data_folder + '/' + args["date"]
all_imgs = os.listdir(current_img_folder)
processed_data = []

for input_img in all_imgs:
	PATH = current_img_folder + '/' + input_img
	classification = classify_image_mask_usage(
		PATH,
		input_img,
		args["date"],
		person_detection_net,
		face_detection_net,
		mask_model)

	processed_data = processed_data + classification

current_data = pd.read_csv(args['mask_data_csv'])
new_data = pd.DataFrame(processed_data)
all_data = current_data.append(new_data)
all_data.to_csv(args['mask_data_csv'])


