
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-pc", "--person_confidence", type=float, default=0.7,
	help="minimum probability to filter weak person detections")
ap.add_argument("-fc", "--face_confidence", type=float, default=0.7,
	help="minimum probability to filter weak face detections")
ap.add_argument("-i", "--input_folder", required=True,
	help="folder name of input, will be used to re-label all images")
ap.add_argument("-n", "--num_photos", type=int, default=1,
	help="number of photos to save for each video")
args = vars(ap.parse_args())

person_detect_prototxt = 'models/MobileNetSSD_deploy.prototxt.txt'
person_detect_model = 'models/MobileNetSSD_deploy.caffemodel'
person_detect_img_dimensions = (900, 600)

face_detect_prototxt = 'models/deploy.prototxt.txt'
face_detect_model = 'models/res10_300x300_ssd_iter_140000.caffemodel'

print("[INFO] loading object detection model...")
person_detection_net = cv2.dnn.readNetFromCaffe(person_detect_prototxt, person_detect_model)

print("[INFO] loading face detection model...")
face_detection_net = cv2.dnn.readNetFromCaffe(face_detect_prototxt, face_detect_model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

current_wd = os.getcwd()
input_folders = current_wd + '/' + args["input_folder"]
folders = os.listdir(input_folders)
temp_folder = input_folders + '/temp'
processed_images_path = current_wd + '/processed_images'
print("[INFO] creating temp dir for image storage...")

def delete_all_files_in_temp_dir():
	all_files = os.listdir(temp_folder)
	if all_files != None:
		for f in all_files:
			os.remove(temp_folder + '/' + f)

def delete_temp_dir():
	os.rmdir(temp_folder)

if not os.path.isdir(temp_folder):
	os.mkdir(temp_folder)
else:
	delete_all_files_in_temp_dir()

for f in folders:
	all_videos_path = input_folders + '/' + f
	all_videos = os.listdir(all_videos_path)
	for v in all_videos:
		video_path =  all_videos_path + '/' + v
		(W, H) = (None, None)
		vs = cv2.VideoCapture(video_path)
		iter = 0
		print(video_path)

		# clear temp directory
		delete_all_files_in_temp_dir()

		# create dict to store photos and record confidence
		current_photos = []

		while True:
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			# if the frame dimensions are empty, grab them
			if W is None or H is None:
				(H, W) = frame.shape[:2]
			person_blob = cv2.dnn.blobFromImage(cv2.resize(frame, person_detect_img_dimensions),
										 0.007843, (300, 300), 127.5)
			person_detection_net.setInput(person_blob)
			person_detections = person_detection_net.forward()
			for i in np.arange(0, person_detections.shape[2]):
				person_confidence = person_detections[0, 0, i, 2]
				idx = int(person_detections[0, 0, i, 1])
				if CLASSES[idx] == 'person':
					if person_confidence > args["person_confidence"]:
						box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")


						# face detection section

						# trim off bottom part to improve img dimensions 
						endY2 = int(startY + ((endY - startY) / 2))
						crop_img = frame[startY:endY2, startX:endX]
						(face_h, face_w) = crop_img.shape[:2]

						face_blob = cv2.dnn.blobFromImage(cv2.resize(crop_img, (300, 300)), 1.0,
													  (300, 300), (104.0, 177.0, 123.0))
						face_detection_net.setInput(face_blob)
						face_detections = face_detection_net.forward()
						for face_detects in range(0, face_detections.shape[2]):
							face_confidence = face_detections[0, 0, face_detects, 2]
							if face_confidence > args["face_confidence"]:
								face_box = face_detections[0, 0, face_detects, 3:7] * np.array(
									[face_w, face_h, face_w, face_h])
								(face_startX, face_startY, face_endX, face_endY) = face_box.astype("int")
								try:
									# try to make the img just a touch larger around the edges
									face_crop_img = crop_img[face_startY-5:face_endY+5, face_startX-5:face_endX+5]
								except:
									face_crop_img = crop_img[face_startY:face_endY,face_startX:face_endX]

								image_name = str(iter) + '.jpg'
								current_photos.append({
									'person_confidence': person_confidence,
									'face_confidence': face_confidence,
									'image_name': image_name
								})
								cv2.imwrite(temp_folder + '/' + image_name, face_crop_img)
								iter += 1
		# take the best photo from the video, move to processed_images
		sorted_photos_list = sorted(current_photos, key = lambda i: i['face_confidence'], reverse=True)
		video_name = v.split('.')[0]
		for i in range(0, args['num_photos']):
			image_name = sorted_photos_list[i]['image_name']
			os.rename(temp_folder + '/' + image_name, processed_images_path + '/' + f + '/' + video_name + '_' + image_name)
		delete_all_files_in_temp_dir()


print("[INFO] deleting temp directory...")
delete_all_files_in_temp_dir()
delete_temp_dir()

