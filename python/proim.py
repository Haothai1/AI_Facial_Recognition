#!/usr/bin/env python3

# Initialize Camera
#      ↓
# Load Face Detector and InsightFace models
#      ↓
# Capture a Frame
#      ↓
# Run Face Detection
#      ↓
# Get Bboxes
#      ↓
# Crop faces
#      ↓
# Preprocess crops
#      ↓
# Run Face Recognition
#      ↓
# Compare embeddings

import libcamera # type: ignore
import cv2 # type: ignore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

from picamera2 import Picamera2, Preview # type: ignore
from picamera2.encoders import MJPEGEncoder # type: ignore
from picamera2.outputs import Output # type: ignore
from picamera2.devices import Hailo # type: ignore

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

def extract_faces_from_tensors(data):
    faces_detected = []
    for class_id, detections in enumerate(data):
        for detection in detections:
            y0, x0, y1, x1 = detection[:4]
            bbox = (int(x0 * face_detector_input_res[0]), int(y0 * face_detector_input_res[1]),
                     int(x1 * face_detector_input_res[0]), int(y1 * face_detector_input_res[1]))
            score = detection[4]
            if score > 0.50 and class_id == 1:
                faces_detected.append([bbox, score])
    return faces_detected

def crop_faces_from_frame(frame, faces):
    cropped_faces = []
    for bbox, score in faces:
        cropped_faces.append(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    return cropped_faces

def pre_process_crops(cropped_faces):
    processed_faces = []
    for face in cropped_faces:
        face_resized = cv2.resize(face, (face_recognizer_input_res[0], face_recognizer_input_res[1]))
        processed_faces.append(face_resized)
    return processed_faces

# START HERE !
if __name__ == "__main__":
    # models
    face_detector = Hailo("/usr/share/hailo-models/yolov5s_personface_h8l.hef")
    face_recognizer = Hailo("/home/pi/arcface_mobilefacenet.hef")

    # data about models
    face_detector_input_res = face_detector.get_input_shape()
    face_recognizer_input_res = face_recognizer.get_input_shape()

    # Configure and start Picamera2.
    picam2 = Picamera2()

    main = {'size': (VIDEO_WIDTH, VIDEO_HEIGHT), 'format': 'XRGB8888'}
    lores = {'size': (face_detector_input_res[0], face_detector_input_res[1]), 'format': 'RGB888'}
    controls = {'FrameRate': 30}
    config = picam2.create_preview_configuration(main, lores=lores, controls=controls)
    picam2.configure(config)    

    picam2.start()
    
    while True:
        #frame = picam2.capture_array("lores")
        frame = cv2.imread("faces/input.jpg")
        frame = cv2.resize(frame, (face_detector_input_res[0], face_detector_input_res[1]))
        face_detector_tensors = face_detector.run(frame)
        faces_detected = extract_faces_from_tensors(face_detector_tensors)
        cropped_faces = crop_faces_from_frame(frame, faces_detected)
        processed_faces = pre_process_crops(cropped_faces)
        for index, face in enumerate(processed_faces):
            cv2.imwrite(f"faces/face{index}_image.jpg", face)
            np.save(f"faces/face{index}_emb.npy", face_recognizer.run(face))
            print("saved!")
        