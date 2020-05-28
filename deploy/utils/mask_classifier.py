import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

face_detect_confidence = 0.75
person_detect_confidence = 0.80
mask_scaler = (1-0.21)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


def classify_image_mask_usage(img_hard_path, image_name, date_string, person_detection_net, face_detection_net, mask_model):
    results = []
    faces_detected = 0
    masks = 0

    frame = cv2.imread(img_hard_path)
    try:
        (h, w) = frame.shape[:2]
    except:
        print('{} failed to be read'.format(img_hard_path))
        return results

    #begin person detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    person_detection_net.setInput(blob)
    person_detections = person_detection_net.forward()

    for i in np.arange(0, person_detections.shape[2]):

        person_confidence = person_detections[0, 0, i, 2]
        if person_confidence > person_detect_confidence:
            idx = int(person_detections[0, 0, i, 1])
            if CLASSES[idx] == 'person':
                box = person_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startY = startY - 10 if startY - 10 > 0 else startY
                endY2 = int(startY + ((endY - startY) / 2))
                crop_img = frame[startY:endY2, startX:endX]
                (face_h, face_w) = crop_img.shape[:2]

                # begin face detection
                face_blob = cv2.dnn.blobFromImage(cv2.resize(crop_img, (600, 600)), 1.0,
                                                  (300, 300), (104.0, 177.0, 123.0))
                face_detection_net.setInput(face_blob)
                face_detections = face_detection_net.forward()
                for face_detects in range(0, face_detections.shape[2]):
                    face_confidence = face_detections[0, 0, face_detects, 2]
                    if face_confidence > face_detect_confidence:

                        faces_detected += 1

                        face_box = face_detections[0, 0, face_detects, 3:7] * np.array([face_w, face_h, face_w, face_h])
                        (face_startX, face_startY, face_endX, face_endY) = face_box.astype("int")

                        face_crop_img = crop_img[face_startY:face_endY, face_startX:face_endX]

                        face_crop_img = cv2.resize(face_crop_img, (96, 96))
                        face_crop_img = face_crop_img.astype("float") / 255.0
                        face_crop_img = img_to_array(face_crop_img)
                        face_crop_img = np.expand_dims(face_crop_img, axis=0)

                        # mask prediction
                        (no_mask, mask) = mask_model.predict(face_crop_img)[0]
                        label = "mask" if mask > mask_scaler else "No Mask"
                        if label == 'mask':
                            masks += 1
    results.append({
        'date': date_string,
        'file_name': image_name,
        'faces_detected': faces_detected,
        'masks': masks,
        'no_masks': faces_detected - masks
    })
    return results
