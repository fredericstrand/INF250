#!/usr/bin/env python3
"""
Simplified Drone Defense HUD - Teal Theme
Fixed and streamlined
"""

import cv2
import cvzone
from ultralytics import YOLO
import math
import time
from collections import deque
import numpy as np


class SimpleDroneHUD:
    def __init__(self, model_path="best_yolov8.pt"):
        self.model = YOLO(model_path)
        
        # State
        self.heading = 0
        self.drone_count = 0
        self.shots_fired = 0
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        # Teal color scheme (BGR format)
        self.teal = (128, 128, 0)         # Dark teal
        self.teal_light = (255, 255, 0)   # Cyan/light teal
        self.teal_bright = (255, 200, 0)  # Bright teal
        self.red = (0, 0, 255)            # Red for alerts
        self.white = (255, 255, 255)      # White text
        
        # UI settings
        self.ui_alpha = 0.7
    
    def fire_shot(self):
        self.shots_fired += 1
    
    def calculate_fps(self):
        self.frame_times.append(time.time())
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
    
    def draw_compass(self, frame):
        """Simple compass - top left"""
        cx, cy = 60, 60
        r = 40
        
        # Background
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + 5, (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.ui_alpha, frame, 1 - self.ui_alpha, 0, frame)
        
        # Ring
        cv2.circle(frame, (cx, cy), r, self.teal_light, 2)
        
        # North marker
        rad = math.radians(-self.heading)
        nx = int(cx + (r - 10) * math.sin(rad))
        ny = int(cy - (r - 10) * math.cos(rad))
        cv2.putText(frame, 'N', (nx - 6, ny + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.teal_bright, 2)
        
        # Arrow
        cv2.arrowedLine(frame, (cx, cy), (cx, cy - 15),
                       self.teal_light, 2, tipLength=0.4)
        
        # Heading number
        cvzone.putTextRect(frame, f"{int(self.heading)}", 
                          (cx - 12, cy + r + 18),
                          scale=0.5, thickness=1, offset=3,
                          colorR=self.teal, colorT=self.white)
        
        return frame
    
    def draw_status_bar(self, frame):
        """Simple status bar at top"""
        h, w = frame.shape[:2]
        
        # Background bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (15, 15, 15), -1)
        cv2.addWeighted(overlay, self.ui_alpha, frame, 1 - self.ui_alpha, 0, frame)
        
        # Border
        cv2.line(frame, (0, 35), (w, 35), self.teal_light, 2)
        
        # Info elements
        y = 22
        
        # FPS - left
        cvzone.putTextRect(frame, f"FPS: {self.fps:.0f}", (20, y),
                          scale=0.6, thickness=1, offset=4,
                          colorR=self.teal, colorT=self.white)
        
        # Targets - center
        target_color = self.teal if self.drone_count == 0 else self.red
        cvzone.putTextRect(frame, f"TARGETS: {self.drone_count}", 
                          (w//2 - 50, y),
                          scale=0.6, thickness=1, offset=4,
                          colorR=target_color, colorT=self.white)
        
        # Shots - right
        cvzone.putTextRect(frame, f"SHOTS: {self.shots_fired}", 
                          (w - 120, y),
                          scale=0.6, thickness=1, offset=4,
                          colorR=self.teal_bright, colorT=(0, 0, 0))
        
        return frame
    
    def draw_detection(self, frame, x1, y1, x2, y2, confidence, drone_id):
        """Simple detection box"""
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Corner brackets
        cvzone.cornerRect(frame, (x1, y1, width, height),
                         l=15, t=2, rt=1,
                         colorR=self.red,
                         colorC=self.teal_light)
        
        # Label
        label = f"D{drone_id} {confidence:.2f}"
        cvzone.putTextRect(frame, label, (x1, y1 - 5),
                          scale=0.5, thickness=1, offset=3,
                          colorR=self.red, colorT=self.white)
        
        # Center crosshair
        cv2.drawMarker(frame, (cx, cy), self.teal_light,
                      cv2.MARKER_CROSS, 12, 2)
        
        return frame
    
    def process_frame(self, frame, conf_threshold=0.25):
        """Process frame and add HUD"""
        self.calculate_fps()
        
        # Detect
        results = self.model(frame, conf=conf_threshold, verbose=False)
        self.drone_count = len(results[0].boxes)
        
        # Draw detections
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            frame = self.draw_detection(frame, x1, y1, x2, y2, conf, i + 1)
        
        # Draw HUD
        frame = self.draw_status_bar(frame)
        frame = self.draw_compass(frame)
        
        return frame


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Drone HUD')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--model', type=str, default='best_yolov8.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    
    print("=" * 50)
    print("DRONE DEFENSE HUD")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Screenshot")
    print("  f - Fire shot")
    print("  r - Reset shots")
    print("=" * 50)
    
    # Initialize
    hud = SimpleDroneHUD(model_path=args.model)
    
    # Open video
    try:
        source = int(args.source)
    except:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.source}")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No more frames")
                break
            
            # Simulate compass
            hud.heading = (hud.heading + 0.5) % 360
            
            # Process
            hud_frame = hud.process_frame(frame, conf_threshold=args.conf)
            frame_count += 1
            
            # Display
            cv2.imshow('Drone HUD', hud_frame)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('s'):
                filename = f"hud_{frame_count}.jpg"
                cv2.imwrite(filename, hud_frame)
                print(f"Saved: {filename}")
            elif key == ord('f'):
                hud.fire_shot()
                print(f"Shot! Total: {hud.shots_fired}")
            elif key == ord('r'):
                hud.shots_fired = 0
                print("Shots reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 50)
        print(f"Frames: {frame_count}")
        print(f"Shots: {hud.shots_fired}")
        print(f"FPS: {hud.fps:.1f}")
        print("=" * 50)


if __name__ == "__main__":
    main()