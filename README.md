# Smart City Traffic Monitoring & Automatic Number Plate Recognition (ANPR) System

## 🎥 Demo Video

[![Watch Demo](https://img.youtube.com/vi/NAihJOi8sNA/0.jpg)](https://www.youtube.com/watch?v=NAihJOi8sNA)

## 🎥 Demo Video

[▶️ Watch Video](https://www.youtube.com/watch?v=NAihJOi8sNA)

This project demonstrates a real-time AI-based solution for detecting, tracking, counting vehicles, and extracting license plate numbers from video streams. It is designed as part of a smart city initiative to improve traffic monitoring, law enforcement, and automation.

## Key Features:
• Vehicle Detection using YOLO (Car, Motorcycle, Truck)  
• Multi-object Tracking using ByteTrack  
• Accurate Vehicle Counting with optimized line-crossing logic  
• License Plate Detection using a custom YOLO model  
• OCR-based Plate Recognition using EasyOCR  
• Advanced Image Preprocessing for improved text accuracy  
• Multi-frame voting for stable plate detection  
• Data logging (Vehicle Type, Track ID, Plate Number, Timestamp)  

## 🏗️ System Architecture

```mermaid
flowchart TD

A[Input Video / Camera Stream] --> B[Frame Extraction (OpenCV)]

B --> C[Vehicle Detection (YOLOv8)]
C --> D[Object Tracking (ByteTrack)]

D --> E[Vehicle Counting Logic]
E --> F[Vehicle Records Storage]

B --> G[License Plate Detection (YOLOv8)]
G --> H[Plate Cropping]

H --> I[Image Preprocessing]
I --> J[OCR (EasyOCR)]

J --> K[Plate Text Output]

D --> L[Vehicle Bounding Boxes]
K --> M[Plate- Vehicle Association]

L --> M
M --> N[Multi-frame Voting]

N --> O[Final Plate Number]

O --> F

F --> P[CSV / Database Storage]

P --> Q[Output Video with Annotations]

## Tech Stack:
Python | OpenCV | YOLOv8 | ByteTrack | EasyOCR | NumPy

## Applications:
• Smart City Traffic Management  
• Automated Toll Systems  
• Law Enforcement & Surveillance  
• Parking Management Systems  

### This system demonstrates how computer vision can be applied to real-world problems, enabling intelligent and automated traffic analysis.

## Future Improvements:
• Real-time dashboard   
• Super-resolution for clearer license plates  
• Integration with databases and cloud systems

