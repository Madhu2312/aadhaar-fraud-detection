import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import firebase_admin
from firebase_admin import credentials, db

# -------------------- Firebase Setup --------------------
def initialize_firebase():
    """
    Initialize Firebase app with your service account and database URL.
    """
    cred = credentials.Certificate("path/to/your/serviceAccountKey.json")  # <-- Replace with your path
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://your-database-name.firebaseio.com/'   # <-- Replace with your database URL
    })


# -------------------- YOLO Setup --------------------
def load_yolo_model(cfg_path, weights_path, names_path):
    """
    Load YOLO model from config, weights and class names files.
    """
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


# -------------------- Object Detection --------------------
def detect_objects(img, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    """
    Perform YOLO detection on an image.
    """
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    result_boxes = [boxes[i[0]] for i in indices]
    result_class_ids = [class_ids[i[0]] for i in indices]
    result_confidences = [confidences[i[0]] for i in indices]

    return result_boxes, result_class_ids, result_confidences


# -------------------- QR Code Detection --------------------
def detect_qr_codes(img):
    """
    Detect and decode QR codes in an image using pyzbar.
    """
    qr_codes = pyzbar.decode(img)
    decoded_data = []
    for qr in qr_codes:
        x, y , w, h = qr.rect
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        qr_data = qr.data.decode('utf-8')
        decoded_data.append(qr_data)
        cv2.putText(img, qr_data, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return decoded_data


# -------------------- Firebase Upload --------------------
def upload_to_firebase(node_name, data):
    """
    Upload data to Firebase Realtime Database.
    """
    ref = db.reference(node_name)
    ref.push(data)


# -------------------- Main Processing --------------------
def main():
    # Initialize Firebase
    initialize_firebase()

    # Load YOLO model
    net, classes, output_layers = load_yolo_model(
        cfg_path="yolov3.cfg",           # <-- Replace with your YOLO config
        weights_path="yolov3.weights",   # <-- Replace with your YOLO weights
        names_path="coco.names"          # <-- Replace with your class names
    )

    # Open camera or video
    cap = cv2.VideoCapture(0)  # 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect YOLO objects
        boxes, class_ids, confidences = detect_objects(frame, net, output_layers)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Detect QR codes
        qr_data_list = detect_qr_codes(frame)
        for qr_data in qr_data_list:
            print("QR Detected:", qr_data)
            upload_to_firebase("Detected_QR_Codes", qr_data)

        cv2.imshow("YOLO + QR Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
