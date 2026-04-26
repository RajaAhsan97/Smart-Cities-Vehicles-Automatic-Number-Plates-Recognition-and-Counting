from ultralytics import YOLO           # for YOLO model
import supervision as sv               # for ByteTracker
import cv2                             # for image processing

import util
from util import (preprocess_plate, read_plate_text, count_vehicles, associate_plate_with_vehicle,
                  draw_counting_lines, display_vehicles_counts, save_records)

# vehicles detection model
vehicles_detection_model = YOLO('yolov8n.pt')
# license plate model
License_plate_model = YOLO('license_plate_detector.pt')

# counting logic improved
# output video: videoplayback_counting_&platenumber_v1.mp4
cap = cv2.VideoCapture('YTDown.com_YouTube_License-Plate-Detection-Test_Media_FsGPxhidwGg_001_1080p.mp4')

# get width, height and fps of input video
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# create output Video Writer instance
out = cv2.VideoWriter(
    'videoplayback_counting_&platenumber_CL_improved_v2.mp4',
    cv2.VideoWriter_fourcc(*'MP4V'),
    fps,
    (w, h)
)

# Create instance of Tracker
tracker = sv.ByteTrack(track_activation_threshold=0.6,
                       minimum_matching_threshold=0.85,
                       lost_track_buffer=30,
                       frame_rate=fps)

# Set line position for counting logic
line_position = 700
start_line = (0, line_position)
end_line = (w, line_position)


zone_thickness = int(0.08 * h)   # 8% of frame height
upper_line = line_position - zone_thickness//2
lower_line = line_position + zone_thickness//2

# setting required classes
# Car --> 2
# Motorcycle --> 3
# Truck --> 7
vehicles_classes = [2, 3, 7]

# Colors for vehicles
CLASS_COLORS = {
    2: (0, 255, 0),    # Green for Car
    3: (255, 0, 0),    # Blue for Motorcycle
    7: (0, 0, 255)     # Red for Truck
}

frame_count = 0

# Loop for iterating over every frame of input video
while True:
    # Read frame (one by one)
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # create copy of frame for annotation
    frame_copy = frame.copy()

    # =========================
    # VEHICLE DETECTION
    # =========================
    results = vehicles_detection_model.predict(frame, conf=0.5, classes=vehicles_classes)[0]

    # convert YOLO predictions to Supervision Library format - for tracking
    detections = sv.Detections.from_ultralytics(results)
    # Track detection --> Assigning unique IDs to every vehicel detections
    detections = tracker.update_with_detections(detections)

    # get Bounding boxes i.e (x1,y1,x2,y2) of each detections
    bboxes = results.boxes.xyxy.cpu()
    # get class label of each detections i.e Car-->2, motorcycle-->3, Truck-->7
    classes = results.boxes.cls.int().cpu().tolist()
    # get tracking IDs of each detections
    track_ids = detections.tracker_id.tolist()

    vehicle_boxes = {}

    # loop for iterating over each bounding boxes, class and traking ids
    # for counting logic and drawing boxes on detections
    for bbox, class_id, track_id in zip(bboxes, classes, track_ids):
        # map x1, y2, x2, y2 coordinates to integer values
        x1, y1, x2, y2 = map(int, bbox)
        
        # get colors by class ID
        color = CLASS_COLORS.get(class_id)
        # draw bounding boxes
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

        # storing coordinates by tracking id
        vehicle_boxes[track_id] = (x1, y1, x2, y2)

        # count vehicles
        count_vehicles(x1, y1, x2, y2, class_id, track_id, upper_line, lower_line)
    
    # =========================
    # LICENSE PLATE DETECTION
    # =========================
    plate_results = License_plate_model.predict(frame, conf=0.4)[0]

    plate_detections = []

    if plate_results.boxes is not None:
        for pb in plate_results.boxes.xyxy.cpu():
            # map plate detection coordinate to integers
            px1, py1, px2, py2 = map(int, pb)

            # Skip plates that are too small for reliable OCR
            plate_w = px2 - px1
            plate_h = py2 - py1
            if plate_w < 30 or plate_h < 10:
                continue

            # Crop License plate from frame --> for reading text using OCR
            plate_crop = frame[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            # Preprocess plate
            thresh = preprocess_plate(plate_crop)

            # Read plate text
            plate_text = read_plate_text(thresh)

            # compute centroid of license plate
            plate_center = ((px1 + px2) // 2, (py1 + py2) // 2)

            plate_detections.append({
                "bbox": (px1, py1, px2, py2),
                "center": plate_center,
                "text": plate_text
            })


    frame_copy = associate_plate_with_vehicle(frame_copy, vehicle_boxes, plate_detections)
 
    frame_copy = draw_counting_lines(frame_copy,w, upper_line, lower_line)

    frame_copy = display_vehicles_counts(frame_copy, CLASS_COLORS)


    out.write(frame)

cap.release()
out.release()


save_records()