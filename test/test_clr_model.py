# goal: get fed in an image of people (can be multiple!) walking down the street
# draw bounding box around person
# take bounding box area -> do face detection
# on the face detection determine mask or no mask
# spit out image with mask or no mask on bounding box label of person
# print out number of people detected and if they're wearing mask or not

import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model

mask_no_mask_model = 'models/clr_minigooglenet_96epoch_96x96_32BS.model'


person_detect_prototxt = 'models/MobileNetSSD_deploy.prototxt.txt'
person_detect_model = 'models/MobileNetSSD_deploy.caffemodel'
person_detect_confidence = 0.80

face_detect_prototxt = 'models/deploy.prototxt.txt'
face_detect_model = 'models/res10_300x300_ssd_iter_140000.caffemodel'
face_detect_confidence = 0.75



# person detection
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading object detection model...")
net = cv2.dnn.readNetFromCaffe(person_detect_prototxt, person_detect_model)

print("[INFO] loading face detection model...")
net_face = cv2.dnn.readNetFromCaffe(face_detect_prototxt, face_detect_model)

print("[INFO] loading mask detection model...")
mask_model = load_model(mask_no_mask_model)

current_wd = os.getcwd()
test_imgs = current_wd + '/test_images'
test_imgs_processed = current_wd + '/processed_images/'
all_test_imgs = os.listdir(test_imgs)
all_imgs_with_person = []

img_number = 0

for input_img in all_test_imgs:
    PATH = test_imgs + '/' + input_img
    img_number += 1

    frame = cv2.imread(PATH)
    try:
        (h, w) = frame.shape[:2]
    except:
        print(PATH)
        continue

    # need to modify re-sizing
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    people = 0

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        if confidence > person_detect_confidence:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == 'person':
                people += 1
                all_imgs_with_person.append(input_img)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startY = startY - 10 if  startY - 10 > 0 else startY
                endY2 = int(startY + ((endY-startY)/2))
                crop_img = frame[startY:endY2, startX:endX]
                (face_h, face_w) = crop_img.shape[:2]

                # cut crop img in half height wise
                # print(startY)
                # print(endY)
                # print(face_h)



                cv2.imshow("Frame", cv2.resize(crop_img, (600, 600)))
                key = cv2.waitKey() & 0xFF

                face_blob = cv2.dnn.blobFromImage(cv2.resize(crop_img, (600, 600)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0))
                net_face.setInput(face_blob)
                face_detections = net_face.forward()
                for face_detects in range(0, face_detections.shape[2]):
                    face_confidence = face_detections[0, 0, face_detects, 2]
                    if face_confidence > face_detect_confidence:
                        print(face_confidence)
                        face_box = face_detections[0, 0, face_detects, 3:7] * np.array([face_w, face_h, face_w, face_h])
                        (face_startX, face_startY, face_endX, face_endY) = face_box.astype("int")

                        face_crop_img = crop_img[face_startY:face_endY, face_startX:face_endX]

                        face_crop_img = cv2.resize(face_crop_img, (96, 96))
                        face_crop_img = face_crop_img.astype("float") / 255.0
                        face_crop_img = img_to_array(face_crop_img)
                        face_crop_img = np.expand_dims(face_crop_img, axis=0)
                        (no_mask, mask) = mask_model.predict(face_crop_img)[0]
                        print(mask)
                        print(no_mask)
                        label = "mask" if mask > (1-0.21) else "No Mask"
                        proba = mask if mask else mask + 0.45

                        # run in through object detect model right here


                        # draw the prediction on the frame
                        # label = "{}: {:.2f}%".format(CLASSES[idx],
                        #                              confidence * 100)
                        label = "{}: {:.2f}%".format(label, proba * 100)
                        cv2.rectangle(frame, (startX +  face_startX, startY + face_startY), (startX + face_endX, startY + face_endY),
                                      COLORS[idx], 2)
                        y = face_startY - 15 if face_startY - 15 > 15 else face_startY + 15
                        cv2.putText(frame, label, (face_startX + startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        # try:
                        cv2.imshow("Frame", frame)
                        # except:
                        #     print('didnt work')
                        #     cv2.imshow("Frame", crop_img)
                        key = cv2.waitKey() & 0xFF
                
    cv2.imwrite(test_imgs_processed + '/' + str(img_number) + '.jpg', frame)

    print('{} people detected'.format(str(people)))

for i in all_test_imgs:
    if i not in all_imgs_with_person:
        print('person not detected for {}'.format(i))