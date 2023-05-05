import cv2
from ultralytics import YOLO
from sort import *
import math

############# ĐẾM XE VÀ ĐO TỐC ĐỘ #############

############# MADE BY: PHẠM MINH NHẬT #########

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ] # CÁC CLASS TRONG YOLO

model = YOLO('yolov8s.pt') #MÔ HÌNH YOLO

## XÉT TỌA ĐỘ TRONG FRAME ##
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

## ĐỌC FRAME ##
cap = cv2.VideoCapture(r'C:\Users\LENOVO\PycharmProjects\Object Detection\Object-Detection-101\Videos\veh2.mp4')


# print(class_list)

## SORT TRACKING THEO DÕI ĐỐI TƯỢNG ##

tracker = Sort()

## KHAI BÁO MẢNG, CÁC BIẾN CẦN THIẾT ##
count = 0
cy1 = 322
cy2 = 368
offset = 6

vh_down={}
vh_downspeed={}
vh_up={}
vh_upspeed={}
count_vhdown = []
count_vhup = []

## MAIN ##
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    list = np.empty((0, 5))
    #   print(results)
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass =="car":
                currentArray = np.array([x1, y1, x2, y2, conf])
                # print(currentArray)
                list = np.vstack((list, currentArray))

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
        w, h = x4 - x3, y4 - y3
        cx, cy = x3 + w // 2, y3 + h // 2
        # print("cy", cy)
        if cy1<(cy+offset) and cy1>(cy-offset):
            vh_down[id]=cy
            vh_downspeed[id]=time.time()
            print(vh_downspeed)
        if id in vh_down:
           if cy2 < (cy + offset) and cy2 > (cy - offset):
              elapsed_timedown = time.time() - vh_downspeed[id]

              if count_vhdown.count(id)==0:
                count_vhdown.append(id)
                distance=10
                speed_down_ms= distance/elapsed_timedown
                speed_down_kmh=speed_down_ms*3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, str(int(speed_down_kmh))+'Km/h', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if cy2<(cy+offset) and cy2>(cy-offset):
            vh_up[id]=cy
            vh_upspeed[id] = time.time()
        if id in vh_up:
           elapsed_timeup = time.time() - vh_upspeed[id]
           if cy1 < (cy + offset) and cy1 > (cy - offset):

              if count_vhup.count(id)==0:
                count_vhup.append(id)
                distance = 10
                speed_up_ms = distance / elapsed_timeup
                speed_up_kmh = speed_up_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, str(int(speed_up_kmh)) + 'Km/h', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 255), 2)
    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
    cv2.putText(frame, 'line1', (274, 318), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 1)
    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.putText(frame, 'line2', (181, 363), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 1)
    cv2.putText(frame, f'so xe lan ben phai {len(count_vhdown)}', (470, 58), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 255, 255), 2)
    cv2.putText(frame, f'so xe lan ben trai {len(count_vhup)}', (470, 100), cv2.FONT_HERSHEY_SIMPLEX , 1.5, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
