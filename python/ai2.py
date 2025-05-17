#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
import socket
import struct
import random
from sklearn.metrics.pairwise import cosine_similarity
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import Output
from picamera2.devices import Hailo

# Environment setup
os.environ["HAILO_DEFAULT_STREAM_TIMEOUT_MS"] = "10000"
os.environ["DISPLAY"] = ":0"

# Configuration
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 800
SERVER_IP = "192.168.1.136"
SERVER_PORT = 5005
SAVE_INTERVAL = 5  # Save faces every 5 seconds
FACES_DIR = "faces"

# Create faces directory if not exists
os.makedirs(FACES_DIR, exist_ok=True)

# Global variables
faces_detected = []
face_detector_input_res = (640, 640)
face_recognizer_input_res = (112, 112)
last_save_time = time.time()

class SocketOutput(Output):
    def __init__(self, sock=None):
        self.sock = sock
        self.last_success = time.time()
        
    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=None):
        if self.sock:
            try:
                # Only try to send if we had a recent successful connection
                if time.time() - self.last_success < 10:  # 10 second grace period
                    self.sock.settimeout(2)  # Shorter timeout
                    self.sock.sendall(struct.pack(">I", len(frame)))
                    self.sock.sendall(frame)
                    self.last_success = time.time()
            except Exception as e:
                # Only print errors occasionally to reduce spam
                if random.random() < 0.1:  # 10% chance to print error
                    print(f"Socket send failed: {str(e)[:50]}")

def save_detected_faces(faces, frame):
    """Save detected faces to disk with timestamp"""
    global last_save_time
    current_time = time.time()
    
    if current_time - last_save_time >= SAVE_INTERVAL and faces:
        timestamp = int(current_time)
        for i, (bbox, score) in enumerate(faces):
            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if face_img.size > 0:
                filename = f"{FACES_DIR}/face_{timestamp}_{i}_score_{int(score*100)}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                print(f"Saved {filename}")
        last_save_time = current_time

def extract_faces_from_tensors(data):
    global face_detector_input_res, faces_detected
    faces_detected = []
    for class_id, detections in enumerate(data):
        for detection in detections:
            y0, x0, y1, x1 = detection[:4]
            bbox = (int(x0 * face_detector_input_res[0]), 
                    int(y0 * face_detector_input_res[1]),
                    int(x1 * face_detector_input_res[0]), 
                    int(y1 * face_detector_input_res[1]))
            score = detection[4]
            if score > 0.50 and class_id == 1:
                faces_detected.append([bbox, score])
    return faces_detected

def crop_faces_from_frame(frame, faces):
    return [frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox, _ in faces]

def pre_process_crops(cropped_faces):
    global face_recognizer_input_res
    processed_faces = []
    for face in cropped_faces:
        if face is None or face.size == 0:
            continue
        face_resized = cv2.resize(face, face_recognizer_input_res)
        processed_faces.append(face_resized)
    return processed_faces

def draw_objects(request):
    global faces_detected, processed_faces, joeys_embedding, face_recognizer
    if faces_detected:
        with MappedArray(request, "main") as m:
            for i, (bbox, score) in enumerate(faces_detected):
                x0, y0, x1, y1 = bbox
                # Get the similarity score for this face
                if i < len(processed_faces):
                    face = processed_faces[i]
                    face_embedding = face_recognizer.run(face)
                    similarity = cosine_similarity([face_embedding], [joeys_embedding])[0][0]
                    label = f"Det: %{int(score * 100)} Sim: {similarity:.2f}"
                else:
                    label = f"Det: %{int(score * 100)}"
                cv2.rectangle(m.array, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(m.array, label, (x0 + 5, y0 + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    # Initialize camera
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "XRGB8888"},
            lores={"size": face_detector_input_res, "format": "RGB888"},
            controls={"FrameRate": 15}
        )
        picam2.configure(config)
        picam2.start_preview(Preview.QTGL, x=100, y=100, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        picam2.start()
        time.sleep(2)
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        exit(1)

    # Initialize models
    try:
        face_detector = Hailo("/usr/share/hailo-models/yolov5s_personface_h8l.hef")
        face_recognizer = Hailo("/home/pi/arcface_mobilefacenet.hef")
        face_detector_input_res = face_detector.get_input_shape()[:2]
        face_recognizer_input_res = face_recognizer.get_input_shape()[:2]
        print(f"Models loaded - Detector: {face_detector_input_res}, Recognizer: {face_recognizer_input_res}")
    except Exception as e:
        print(f"Model loading failed: {e}")
        picam2.stop()
        exit(1)

    # Network connection
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((SERVER_IP, SERVER_PORT))
        print("Network connection established")
    except Exception as e:
        print(f"Network connection failed: {e}")
        print("Continuing without network...")

    # Main processing loop
    try:
        socket_output = SocketOutput(sock)
        encoder = MJPEGEncoder()
        picam2.start_encoder(encoder, socket_output)
        
        joeys_embedding = np.load(f"{FACES_DIR}/face0_emb.npy")
        
        while True:
            start_time = time.time()
            
            # Capture and process frame
            frame = picam2.capture_array("lores")
            face_detector_tensors = face_detector.run(frame)
            faces_detected = extract_faces_from_tensors(face_detector_tensors)
            
            # Save detected faces periodically
            save_detected_faces(faces_detected, frame)
            
            # Recognition pipeline
            cropped_faces = crop_faces_from_frame(frame, faces_detected)
            processed_faces = pre_process_crops(cropped_faces)
            
            for index, face in enumerate(processed_faces):
                face_embedding = face_recognizer.run(face)
                similarity = cosine_similarity([face_embedding], [joeys_embedding])[0][0]
                print(f"[{index}] similarity: {similarity:.4f}")
            
            # Draw bounding boxes
            request = picam2.capture_request()
            draw_objects(request)
            request.release()
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < 1/30:
                time.sleep((1/30) - elapsed)
                
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if sock: sock.close()
        picam2.stop()
