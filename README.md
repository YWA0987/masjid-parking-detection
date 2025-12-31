# Masjid Parking Detection System
AI-powered parking availability monitor using computer vision to help reduce congestion in parking lots. Camera-based system to estimate open parking spaces in a masjid parking lot
using computer vision (YOLOv8 AI) and predefined parking spot coordinates.

## 🎯 Overview

Parking spot detection system that:
- Monitors parking lot with camera
- Uses YOLOv8 AI to detect vehicles
- Counts available parking spaces
- Displays availability on digital sign

**Problem:** Parking congestion during prayer times/jumuah causes traffic backup and frustration

**Solution:** Show drivers availability BEFORE they enter the lot

## 🚧 Current Status

[x] System architecture and core approach finalized

[x] Coordinate-based parking spot method agreed upon

[x] YOLO inference verified on static images (vehicle detection)

[x] Core detection + occupancy logic being drafted

[x] Display output planned for later phase

## ✨ Features
- Near real-time vehicle detection using YOLOv8
- Coordinate-based spot tracking (handles any camera angle)
- Planned: anti-flicker smoothing (to reduce detection jitter)
- Fail-safe status handling
- Privacy-focused (no video recording)
- Works offline after setup

## 🛠 How It Works
(Section will be added once the core detection pipeline is implemented.)
