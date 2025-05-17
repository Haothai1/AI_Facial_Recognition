# Facial Recognition System Setup

## Prerequisites
- Raspberry Pi with Raspberry Pi OS
- Hailo-8 AI accelerator
- Camera module (compatible with Picamera2)
- Python 3.11 or later

## Installation

1. Install the Raspberry Pi OS.

2. Install the Hailo-RT for Hailo 8 AI accelerator.

3. Install required system packages:
```bash
sudo apt update
sudo apt install python3
```

4. Create and activate a virtual environment:
```bash
python3 -m venv aienv
source aienv/bin/activate
```

5. Install required Python packages:
```bash
pip install numpy opencv-python scikit-learn
```

6. Install Hailo dependencies:
```bash
sudo apt install hailo-python
```

## Configuration

1. Create a `faces` directory in the same location as ai2.py:
```bash
mkdir faces
```

2. Update the following variables in ai2.py if needed:
- `SERVER_IP`: Your server IP address (default: "192.168.1.136")
- `SERVER_PORT`: Your server port (default: 5005)
- `VIDEO_WIDTH` and `VIDEO_HEIGHT`: Camera resolution (default: 800x800)

## Running the Program

1. Activate the virtual environment:
```bash
source aienv/bin/activate
```

2. Run the program:
```bash
python ai2.py
```

3. First Run:
- The program will look for a reference face in `faces/face0_emb.npy`
- If not found, it will wait for you to look at the camera
- The first face detected will be saved as the reference face

## Features
- Real-time face and person detection
- Face recognition with similarity scoring
- Automatic face image saving
- Optional network streaming
- Visual display with bounding boxes and scores

## Troubleshooting

1. Camera Issues:
- Ensure camera is properly connected
- Check camera permissions
- Verify Picamera2 is installed

2. Hailo Issues:
- Verify Hailo-8 is properly connected
- Check Hailo model files exist:
  - `/usr/share/hailo-models/yolov5s_personface_h8l.hef`
  - `/home/pi/arcface_mobilefacenet.hef`

3. Network Issues:
- Program will continue to run locally if network connection fails
- Verify server IP and port if streaming is required

## Stopping the Program
- Press Ctrl+C to stop the program
- The program will clean up resources automatically