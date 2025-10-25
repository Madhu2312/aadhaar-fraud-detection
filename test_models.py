from ultralytics import YOLO
import os

print("Testing YOLO models...")

# Test 1: Try to load yolov8n.pt
try:
    print("1. Testing yolov8n.pt...")
    model1 = YOLO("yolov8n.pt")
    print("    yolov8n.pt loaded successfully")
    print(f"   Model classes: {model1.names}")
except Exception as e:
    print(f"    yolov8n.pt failed: {e}")

# Test 2: Try to load the custom model
try:
    print("2. Testing model.pt...")
    model2 = YOLO("models/model.pt")
    print("    model.pt loaded successfully")
    print(f"   Model classes: {model2.names}")
except Exception as e:
    print(f"    model.pt failed: {e}")

# Test 3: Try to load best.pt
try:
    print("3. Testing best.pt...")
    # Check file size first
    file_size = os.path.getsize("models/best.pt")
    print(f"   File size: {file_size} bytes")
    if file_size < 1000:
        print("    best.pt is too small (likely corrupted)")
    else:
        model3 = YOLO("models/best.pt")
        print("    best.pt loaded successfully")
        print(f"   Model classes: {model3.names}")
except Exception as e:
    print(f"    best.pt failed: {e}")
