import numpy as np
import cv2


prototxt_path = r"C:\Users\upraj\OneDrive\Desktop\Object Detection\models\MobileNetSSD_deploy.caffemodel"
model_path = r"C:\Users\upraj\OneDrive\Desktop\Object Detection\models\MobileNetSSD_deploy.prototxt.txt"

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(prototxt_path, model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, image = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detected_objects = net.forward()

    min_confidence = 0.2
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            if class_index < len(classes):
                upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                label = f"{classes[class_index]}: {confidence:.2f}"
                color = tuple(map(int, colors[class_index]))
                cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), color, 3)
                cv2.putText(image, label, (upper_left_x, upper_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 3)

    cv2.imshow("Detected objects", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
