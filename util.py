import cv2
import easyocr
import numpy as np
import re
import datetime
from collections import defaultdict, Counter
import csv

# =========================
# GLOBAL VARIABLES
# =========================
# TRACKING MEMORY
track_states = {}
track_age = {}
prev_positions = {}
counted_ids = set()
# Initialize vehicle counting variable
vehicle_counts = {2: 0, 3: 0, 7: 0}
# Vehicle types
vehicle_types = {
    2: "Car",
    3: "Motorcycle",
    7: "Truck"
}
plate_history = defaultdict(list)
plate_tracks = {}
# for storing VehicleType, TrackingID, plateNumber and Timestamp
records = {}


# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# draw plate bounding box
def draw_plate_bbox(frame_copy, px1, py1, px2, py2):
    # Draw plate bounding box
    cv2.rectangle(frame_copy, (px1, py1), (px2, py2), (255, 255, 0), 2)
    return frame_copy

# display plate number on top of vehicles bounding box
def display_plate_number(frame_copy, display_text, vx1, vy1):
    #cv2.putText(frame, display_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    text_x = vx1
    text_y = vy1 - 10 if vy1 - 10 > 10 else vy1 + 30

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0   # increase for larger text
    thickness = 2

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

    # Draw WHITE background rectangle
    cv2.rectangle(frame_copy,(text_x, text_y - text_h - 5),(text_x + text_w + 5, text_y + baseline),(255, 255, 255), -1)

    # Draw BLACK text
    cv2.putText(frame_copy,display_text,(text_x, text_y),font,font_scale,(0, 0, 0),thickness,cv2.LINE_AA)

    return frame_copy

def draw_counting_lines(frame_copy, width, upper_line, lower_line):
    # Draw counting line
    cv2.line(frame_copy, (0, upper_line), (width, upper_line), (255,0,0), 2)
    cv2.line(frame_copy, (0, lower_line), (width, lower_line), (0,0,255), 2)

    return frame_copy

def display_vehicles_counts(frame_copy, CLASS_COLORS):
    # Vehicle count display
    cv2.putText(frame_copy, f"Car: {vehicle_counts[2]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLASS_COLORS[2], 2)
    cv2.putText(frame_copy, f"Motorcycle: {vehicle_counts[3]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLASS_COLORS[3], 2)
    cv2.putText(frame_copy, f"Truck: {vehicle_counts[7]}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLASS_COLORS[7], 2)

    return frame_copy


# License Plate preprocessing function
def preprocess_plate(plate_crop):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Trim padding to avoid reading car body instead of plate
    pad = 3
    if gray.shape[0] > pad * 2 and gray.shape[1] > pad * 2:
        gray = gray[pad:-pad, pad:-pad]

    # Upscale 6x with Lanczos interpolation
    upscaled = cv2.resize(gray, (gray.shape[1] * 6, gray.shape[0] * 6),
                          interpolation=cv2.INTER_LANCZOS4)

    # Bilateral Filter
    # Smooth noise while preserve edges
    bilateral = cv2.bilateralFilter(upscaled, d=9, sigmaColor=60, sigmaSpace=60)

    # CLAHE - Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(bilateral)

    # Unsharp Masking
    # Sharpen character edges
    for _ in range(2):
        blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
        enhanced = cv2.addWeighted(enhanced, 2.5, blur, -1.5, 0)

    # Gamma Correction
    # Brighten mid-tones for better contrast
    gamma = 1.6
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).astype(np.uint8)
    brightened = cv2.LUT(enhanced, lut)

    # Otsu's thresholding
    _, binary = cv2.threshold(brightened, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure text is white on black background
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Morphological Close Operation
    # fill gaps inside characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Morphological Open
    # Remove small noise blobs
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return thresh


# OCR reading
def read_plate_text(thresh):
    # Read license plate text
    ocr_results = reader.readtext(
        thresh,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', # read alphanumeric charactors
        min_size=10,
        paragraph=False
    )

    plate_text = ""
    best_conf = 0.0

    for (bbox, text, confidence) in ocr_results:
        print(f"Text: {text} | Confidence: {confidence:.2f}")

        # check if the ocr detection confidence is > 0.5
        # Then store the plate text
        if confidence > 0.5:
            cleaned = text.upper().replace(" ", "")
            cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)

            # Only accept text that looks like a plate (4-8 chars)
            if 4 <= len(cleaned) <= 8 and confidence > best_conf:
                plate_text = cleaned
                best_conf = confidence

    return plate_text


# Vehicles counting logic

def record_vehicles(class_id, track_id):
    global records, plate_tracks
    
    # get vehicel type
    vehicle_type = vehicle_types.get(class_id)
    # get plate number
    plate_number = plate_tracks.get(track_id, "Not Detected")

    # determine timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # store the records i.e. VehicelType, TrackID, PlateNumber, Timestamp
    records[track_id] = {
        "vehicle_type": vehicle_type,
        "track_id": track_id,
        "plate_number": plate_number,
        "timestamp": timestamp
    }


# counting logic
# If centroid of bounding boxes crosses the line
# AND
# tracking ID not in stored IDS
def count_vehicles(x1, y1, x2, y2, class_id, track_id, upper_line, lower_line):
    global track_states, track_age, prev_positions, counted_ids
    global vehicle_counts, records

    # compute centroid of bounding box
    #cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cy = y2
 
    track_age[track_id] = track_age.get(track_id, 0) + 1
    prev_y = prev_positions.get(track_id, cy)
 
    # avoid jitter
    if abs(cy - prev_y) < 2:
        prev_positions[track_id] = cy
        return
 
    # Determine zone
    if cy < upper_line:
        curr_state = "above"
    elif cy > lower_line:
        curr_state = "below"
    else:
        curr_state = "inside"
 
    prev_state = track_states.get(track_id, curr_state)
 
    # True crossing
    if (prev_state == "above" and
        curr_state == "below" and
        track_id not in counted_ids and
        track_age[track_id] > 8):
 
        # store tracking ID
        counted_ids.add(track_id)
        # count vehicle by class ID
        vehicle_counts[class_id] += 1
        # store vehicles records
        record_vehicles(class_id, track_id)
 
    track_states[track_id] = curr_state
    prev_positions[track_id] = cy

# =========================
# ASSOCIATE PLATE WITH VEHICLE
# =========================
# Since Vehicle detection and License plate models are running
# independently on the frame. If vehicle detection model detects 10
# vehicles and License plate model detects 2 license plate. Then, we
# do not know which license plate corresponds to which vehicle. Therefore,
# I have written the logic to associate detected license plates with the
# detected vehicles.
def associate_plate_with_vehicle(frame_copy, vehicle_boxes, plate_detections):
    global plate_history, plate_tracks, records

    # For iterating over each detected vehicles in the frame
    for track_id, (vx1, vy1, vx2, vy2) in vehicle_boxes.items():
        # iterating over each detected license plate in the frame
        for plate in plate_detections:
            # get plate bbox, centroid, text
            px1, py1, px2, py2 = plate["bbox"]
            pcx, pcy = plate["center"]
            plate_text = plate["text"]

            # logic for associting plate with vehicle
            if vx1 < pcx < vx2 and vy1 < pcy < vy2:
                if plate_text:
                    # multi frame voting
                    plate_history[track_id].append(plate_text)

                    # Keep last 15 frames
                    plate_history[track_id] = plate_history[track_id][-15:]

                    # Majority vote for stable plate
                    stable_plate = Counter(
                        plate_history[track_id]
                    ).most_common(1)[0][0]

                    # store stable license plate text
                    plate_tracks[track_id] = stable_plate

                # Update records with plate number
                if track_id in records and plate_tracks.get(track_id):
                    records[track_id]["plate_number"] = plate_tracks[track_id]

                frame_copy = draw_plate_bbox(frame_copy, px1, py1, px2, py2)    

                # Display plate number
                display_text = plate_tracks.get(track_id, "")
                if display_text:
                    frame_copy = display_plate_number(frame_copy, display_text, vx1, vy1)

    return frame_copy

def save_records():
    # =================================
    # Save Detected Vehicles Records 
    # i.e.
    # VehicleType, Track_id, PlateNumber, TimeStamp
    # =================================

    csv_file = "vehicle_records.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # header
        writer.writerow(["vehicle_type", "track_id", "plate_number", "timestamp"])

        # data
        for record in records.values():
            writer.writerow([
                record["vehicle_type"],
                record["track_id"],
                record["plate_number"],
                record["timestamp"]
            ])

    print("CSV saved successfully!")