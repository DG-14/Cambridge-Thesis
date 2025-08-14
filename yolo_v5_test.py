import carla
import numpy as np
import cv2
import torch
import time
import os
import sys

# Import YOLOv5
YOLO_PATH = '../yolov5'  # Path to YOLOv5 repo
sys.path.append(YOLO_PATH)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Load YOLOv5
device = select_device('')
model = attempt_load('yolov5s.pt', map_location=device)  # Use 'yolov5m.pt', etc. if desired
model.eval()

# CARLA settings
IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]
    img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    # Preprocess for YOLO
    img_resized = letterbox(img, new_shape=640)[0]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

    # Draw boxes
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()
        for *xyxy, conf, cls in pred:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show image
    cv2.imshow('YOLO on CARLA', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.load_world('Town03')

blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
camera_bp.set_attribute('fov', '110')

# Spawn vehicle and camera
spawn_point = world.get_map().get_spawn_points()[0]
vehicle_bp = blueprint_library.filter('model3')[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Start streaming
camera.listen(lambda image: process_img(image))

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
