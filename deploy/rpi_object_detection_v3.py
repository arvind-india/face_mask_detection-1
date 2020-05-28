"""
Goal: run on Rpi, triggers taking a photo to then be classified
"""

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import datetime
from datetime import date 
import pandas as pd 
import subprocess

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-oh", "--operating_hours", type=float, default=2.0,
	help="Number of hours to capture data for from starting hour")
ap.add_argument("-sh", "--start_hour", type=int, default=8,
	help="Hour (PST) to start capturing data. Operating time for period from start hour to start hour + hours (t)")
args = vars(ap.parse_args())

PI_CAMERA = True
SECOND_STITCHER = 5
DISPLAY_STALE_FRAME = False
wait_seconds = 5


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] starting video stream...")
if PI_CAMERA:
	vs = VideoStream(usePiCamera=True).start()
else:
	vs = VideoStream(src=0).start()


def grab_non_active_frame(vs):
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	return frame

def trigger_camera_subprocess():
	return subprocess.Popen(['gphoto2', 'capture-image'])

def initialize_dummpy_subprocess():
	return subprocess.Popen(['echo', 'Starting cv person detection and initializing subprocess'])

camera_trigger = initialize_dummpy_subprocess()

if DISPLAY_STALE_FRAME:
	"""
	Allows to show a frame of camera 
	"""
	print('displaying blank frame')
	blank_frame = grab_non_active_frame(vs)
	cv2.imshow("Frame", blank_frame)

outputs_list = []
x_axis_bounding_list = []
this_datetime = datetime.datetime.now() - datetime.timedelta(minutes=5)
def outputs_list_writer(last_datetime, x_axis_bounding_list):
	current_datetime = datetime.datetime.now()	
	if (current_datetime - last_datetime).seconds < (wait_seconds):
		# update number of videos in the last output result of list 
		outputs_list[-1]['num_videos'] = outputs_list[-1]['num_videos'] + 1
		direction = get_direction_from_bounding_list(x_axis_bounding_list)
		outputs_list[-1]['walking_direction'] = direction
		return current_datetime, x_axis_bounding_list
	else:
		print("Started new writer")
		current_datetime_str = str(current_datetime)
		direction = 'unknown'
		x_axis_bounding_list = []
		outputs_list.append({
			'time': current_datetime_str,
			'num_videos': 1,
			'walking_direction': direction
			})
		return current_datetime, x_axis_bounding_list

def get_direction_from_bounding_list(x_axis_bounding_list):
	# in camera, left to right is x = 0 to x = 400
	# right to left is x = 400 to x = 0
	agg_direction = 0
	for i in range(1, len(x_axis_bounding_list)):
		direction = x_axis_bounding_list[i - 1] - x_axis_bounding_list[i]
		agg_direction = agg_direction + direction
	# right to left
	if agg_direction > 0:
		return 'right_to_left'
	if agg_direction < 0:
		return 'left_to_right'
	return 'unknown'


# start with an operating window check
num_captures_upper_bound = 1000
# initialize num_captures to do a operating window check
num_captures = num_captures_upper_bound + 1
has_operated_in_window = False

def light_computation_check_for_operation_window(has_operated_in_window):
	starting_hour = args['start_hour']
	ending_hour = starting_hour + args['operating_hours']
	current_datetime = datetime.datetime.now()
	current_hour_min = current_datetime.hour + (current_datetime.minute / float(60))

	# if IN WINDOW
	if current_hour_min >= starting_hour and current_hour_min <= ending_hour:
		# reset capture count
		has_operated_in_window = True
		return 0, has_operated_in_window
	else:
		# sleep 5 minutes
		print('sleeping at:' + str(current_datetime))
		time.sleep(300)
		return num_captures_upper_bound + 1, has_operated_in_window


time.sleep(2.0)
fps = FPS().start()

try:
	while True:
		if num_captures > num_captures_upper_bound:
			num_captures, has_operated_in_window = light_computation_check_for_operation_window(has_operated_in_window)
			# check if we did a num_captures reset (in sleep mode)
			if num_captures > num_captures_upper_bound:
				# check if we need to do a quick wrap up (when we were in operating window). Helps to save progress
				if has_operated_in_window:
					outputs_df = pd.DataFrame(outputs_list)
					outputs_df.to_csv('output/outputs_and_time.csv')
					has_operated_in_window = False
				continue
		else:
			num_captures += 1

		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)
		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] == 'person':
				if confidence > args["confidence"]:

					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# if camera trigger is None means subprocess is in progress
					if camera_trigger.poll() != None:
						camera_trigger = trigger_camera_subprocess()
						this_datetime, x_axis_bounding_list = outputs_list_writer(this_datetime, x_axis_bounding_list)
						x_axis_bounding_list.append(startX)
					else:
						# do directional analysis here while camera is operating, then update the list
						print('camera busy, analyzing direction')
						x_axis_bounding_list.append(startX)
						direction = get_direction_from_bounding_list(x_axis_bounding_list)
						outputs_list[-1]['walking_direction'] = direction
						time.sleep(0.5)

# stop the timer and display FPS information
except KeyboardInterrupt:
	exit
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

	outputs_df = pd.DataFrame(outputs_list)
	outputs_df.to_csv('output/outputs_and_time.csv')
