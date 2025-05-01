import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import datetime
import numpy as np
import cv2

# Initialize Firebase Admin SDK
cred = credentials.Certificate("path/to/your/serviceAccountKey.json")  # You'll need to provide this
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-project-id.appspot.com'  # Replace with your bucket name
})

# Get Firestore client
db = firestore.client()
bucket = storage.bucket()

def save_face_detection(face_data):
    """
    Save face detection data to Firestore
    face_data should contain:
    - timestamp
    - similarity_score
    - is_recognized (boolean)
    - face_image_url (if uploaded to storage)
    """
    try:
        doc_ref = db.collection('face_detections').document()
        doc_ref.set({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'similarity_score': face_data['similarity_score'],
            'is_recognized': face_data['is_recognized'],
            'face_image_url': face_data.get('face_image_url', ''),
            'alert_sent': face_data.get('alert_sent', False)
        })
        return doc_ref.id
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return None

def upload_face_image(image_data, detection_id):
    """
    Upload face image to Firebase Storage
    Returns the public URL of the uploaded image
    """
    try:
        # Convert image data to bytes if it's a numpy array
        if isinstance(image_data, np.ndarray):
            _, img_encoded = cv2.imencode('.jpg', image_data)
            image_data = img_encoded.tobytes()
        
        # Create a unique filename
        filename = f"faces/{detection_id}.jpg"
        blob = bucket.blob(filename)
        
        # Upload the image
        blob.upload_from_string(image_data, content_type='image/jpeg')
        
        # Make the image publicly accessible
        blob.make_public()
        
        return blob.public_url
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None

def send_alert(detection_id):
    """
    Mark a detection as having an alert sent
    """
    try:
        db.collection('face_detections').document(detection_id).update({
            'alert_sent': True
        })
        return True
    except Exception as e:
        print(f"Error sending alert: {e}")
        return False 