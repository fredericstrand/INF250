#!/usr/bin/env python3

import cv2
import cvzone
from ultralytics import YOLO
import math
import time
from collections import deque
import numpy as np
import torch


class CUDAOptimizedDroneHUD:
    def __init__(self, model_path="best_yolov8.pt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        if self.device == 'cuda':
            dummy_img = torch.zeros((1, 3, 640, 640)).to(self.device)
            print("Warming up GPU...")
            for _ in range(3):
                _ = self.model.predict(dummy_img, verbose=False)
            print("GPU ready!")
        
        self.heading = 0
        self.drone_count = 0
        self.shots_fired = 0
        self.fps = 0
        self.inference_time = 0
        self.frame_times = deque(maxlen=30)
        
        self.teal = (128, 128, 0)
        self.teal_light = (255, 255, 0)
        self.teal_bright = (255, 200, 0)
        self.red = (0, 0, 255)
        self.white = (255, 255, 255)
        
        self.ui_alpha = 0.7
        
        self.use_half_precision = self.device == 'cuda'
        if self.use_half_precision:
            self.model.model.half()
            print("Using FP16 (half precision) for faster inference")
        
        self.frame_skip = 1
        self.last_results = None
    
    def fire_shot(self):
        self.shots_fired += 1
    
    def calculate_fps(self):
        self.frame_times.append(time.time())
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
    
    def draw_compass(self, frame):
        cx, cy = 60, 60
        r = 40
        
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r + 5, (20, 20, 20), -1)
        cv2.addWeighted(overlay, self.ui_alpha, frame, 1 - self.ui_alpha, 0, frame)
        
        cv2.circle(frame, (cx, cy), r, self.teal_light, 2)
        
        rad = math.radians(-self.heading)
        nx = int(cx + (r - 10) * math.sin(rad))
        ny = int(cy - (r - 10) * math.cos(rad))
        cv2.putText(frame, 'N', (nx - 6, ny + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.teal_bright, 2)
        
        cv2.arrowedLine(frame, (cx, cy), (cx, cy - 15),
                       self.teal_light, 2, tipLength=0.4)
        
        cvzone.putTextRect(frame, f"{int(self.heading)}", 
                          (cx - 12, cy + r + 18),
                          scale=0.5, thickness=1, offset=3,
                          colorR=self.teal, colorT=self.white)
        
        return frame
    
    def draw_status_bar(self, frame):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (15, 15, 15), -1)
        cv2.addWeighted(overlay, self.ui_alpha, frame, 1 - self.ui_alpha, 0, frame)
        
        cv2.line(frame, (0, 35), (w, 35), self.teal_light, 2)
        
        y = 22
        
        gpu_text = "GPU" if self.device == 'cuda' else "CPU"
        gpu_color = self.teal_bright if self.device == 'cuda' else (100, 100, 100)
        cvzone.putTextRect(frame, gpu_text, (10, y),
                          scale=0.5, thickness=1, offset=3,
                          colorR=gpu_color, colorT=(0, 0, 0))
        
        cvzone.putTextRect(frame, f"FPS: {self.fps:.0f}", (70, y),
                          scale=0.6, thickness=1, offset=4,
                          colorR=self.teal, colorT=self.white)
        
        cvzone.putTextRect(frame, f"{self.inference_time:.0f}ms", (170, y),
                          scale=0.5, thickness=1, offset=3,
                          colorR=self.teal, colorT=self.white)
        
        target_color = self.teal if self.drone_count == 0 else self.red
        cvzone.putTextRect(frame, f"TARGETS: {self.drone_count}", 
                          (w//2 - 50, y),
                          scale=0.6, thickness=1, offset=4,
                          colorR=target_color, colorT=self.white)
        
        cvzone.putTextRect(frame, f"SHOTS: {self.shots_fired}", 
                          (w - 120, y),
                          scale=0.6, thickness=1, offset=4,
                          colorR=self.teal_bright, colorT=(0, 0, 0))
        
        return frame
    
    def draw_detection(self, frame, x1, y1, x2, y2, confidence, drone_id):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        cvzone.cornerRect(frame, (x1, y1, width, height),
                         l=15, t=2, rt=1,
                         colorR=self.red,
                         colorC=self.teal_light)
        
        label = f"D{drone_id} {confidence:.2f}"
        cvzone.putTextRect(frame, label, (x1, y1 - 5),
                          scale=0.5, thickness=1, offset=3,
                          colorR=self.red, colorT=self.white)
        
        cv2.drawMarker(frame, (cx, cy), self.teal_light,
                      cv2.MARKER_CROSS, 12, 2)
        
        return frame
    
    def process_frame(self, frame, conf_threshold=0.25, frame_count=0):
        self.calculate_fps()
        
        if frame_count % self.frame_skip == 0:
            start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=self.use_half_precision):
                results = self.model(frame, conf=conf_threshold, verbose=False, device=self.device)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            self.inference_time = (time.time() - start_time) * 1000
            self.last_results = results
        else:
            results = self.last_results
        
        self.drone_count = len(results[0].boxes)
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            frame = self.draw_detection(frame, x1, y1, x2, y2, conf, i + 1)
        
        frame = self.draw_status_bar(frame)
        frame = self.draw_compass(frame)
        
        return frame


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA-Optimized Drone HUD')
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--model', type=str, default='best_yolov8.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--frame-skip', type=int, default=1)
    args = parser.parse_args()
    
    print("=" * 60)
    print("CUDA-OPTIMIZED DRONE DEFENSE HUD")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Inference size: {args.imgsz}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Half precision: {args.half}")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: CUDA not available, using CPU")
    
    print("\nControls:")
    print("  q - Quit")
    print("  s - Screenshot")
    print("  f - Fire shot")
    print("  r - Reset shots")
    print("=" * 60)
    
    hud = CUDAOptimizedDroneHUD(model_path=args.model)
    hud.frame_skip = args.frame_skip
    
    if args.half and hud.device == 'cuda':
        hud.use_half_precision = True
        hud.model.model.half()
    
    try:
        source = int(args.source)
    except:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.source}")
        return
    
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No more frames")
                break
            
            hud.heading = (hud.heading + 0.5) % 360
            
            hud_frame = hud.process_frame(frame, conf_threshold=args.conf, frame_count=frame_count)
            frame_count += 1
            
            cv2.imshow('CUDA Drone HUD', hud_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('s'):
                filename = f"hud_cuda_{frame_count}.jpg"
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
        
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Device: {hud.device.upper()}")
        print(f"Frames: {frame_count}")
        print(f"Shots: {hud.shots_fired}")
        print(f"Average FPS: {hud.fps:.1f}")
        print(f"Average Inference: {hud.inference_time:.1f}ms")
        if hud.device == 'cuda':
            print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print("=" * 60)


if __name__ == "__main__":
    main()