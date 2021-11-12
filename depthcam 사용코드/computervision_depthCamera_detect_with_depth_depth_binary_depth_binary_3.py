import cv2
import numpy as np   
import pyrealsense2 as rs
       
import json
import base64
import requests
import time
import threading 
import pymysql

HOST = "database-hanium.cijeogglkrqn.ap-northeast-2.rds.amazonaws.com"
USER = "admin"
PASSWARD = "gksdldma1234!"

sql = 'INSERT INTO depth_container (id, layer) VALUES (%s, %s)'

timestamp = round(time.time())

x_ocr_secret = "ZlBWaUFqdFp4bVFZTUpnTHZucGVwZFVOZUt3cllVY0o="
ocr_invoke_url = "https://f04b1d88f83e41a6b1df8ce399b61fb5.apigw.ntruss.com/custom/v1/9782/eed9df9be97433637c2950146e96ff0956759904a38b1e5e6a9a69a7f6691a68/general"
uuid = "ce3e4c43-4c69-4188-ba2a-8ad24d9615e4"

headers = {
    "X-OCR-SECRET" : x_ocr_secret,
    "Content-Type" : "application/json"
    }

area_label = ''
res_array = []

depth_array = [[], [], [], [], [], [], [], [], []]
tainer_depth = []

class ThreadOCR(threading.Thread):
    
    def __init__(self, color_frame):
        super().__init__()
        self.imgOCR = color_frame[0:480, 260:380].copy()
        
    def run(self):
        try:
            global area_label
            ret, buffer = cv2.imencode('.png', self.imgOCR)
            png_as_text = base64.b64encode(buffer)
            
            data = {
                    "version" : "V1",
                    "requestId" : uuid,
                    "timestamp" : timestamp,
                    "images" : [
                        {
                            "format" : "png",
                            "data" : png_as_text.decode('utf-8'),
                            "name" : "sample_image"
                            }
                        ]
                    }        
    
            data = json.dumps(data)
            response = requests.post(ocr_invoke_url, headers=headers, data=data)
            res = json.loads(response.text)
            res_array = res.get('images')
            for list in res_array[0].get('fields'):
                print(list.get('inferText'))
                area_label = list.get('inferText')
        except TypeError:
            pass

def getContours(color_frame, imgBinary, depth_colormap, depth_frame):
    global area_label
    x_label = 0
    
    thr = ThreadOCR(color_frame)
    thr.daemon = True
    thr.start()
    
    contours, _ = cv2.findContours(imgBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if cv2.contourArea(contour) < areaMin:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        
        center = approx.mean(axis=0)
        center = np.array(center, dtype=np.int16).flatten().tolist()
        if center[0] < 300 or center[0] > 340:
            continue
        
        cv2.drawContours(color_frame, contour, -1, (255, 0, 255), 7)
        cv2.drawContours(depth_colormap, contour, -1, (255, 0, 255), 7)
        x, y, w, h = cv2.boundingRect(approx)
        
        try:
            x_label = int(area_label)
        except ValueError:
            pass
        
        if center[1] < 240:
            y_label = 0
        elif center[1] >= 240:
            y_label = 4
        
        if x_label != 0:
            label = x_label + y_label
        else:
            label = 0
        
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), 
                      
                      (0, 255, 0), 3)
        
        cv2.putText(color_frame, "Center: " + str(center), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(depth_colormap, "Center: " + str(center), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)
        
        cv2.putText(color_frame, "Label: " + str(label), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(depth_colormap, "Label: " + str(label), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)
        if len(depth_array[label]) < 50:
            depth_array[label].append(depth_frame[center[1]][center[0]])

def empty(a):
    pass

cv2.namedWindow('Parameters')
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Area" , "Parameters", 2000, 30000, empty)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not aligned_color_frame:
            continue
        
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())
        color_frame = np.asanyarray(aligned_color_frame.get_data())
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        
        imgGray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.bilateralFilter(imgGray, -1, 10, 5)
        _, imgBinary = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        
        getContours(color_frame, imgBinary, depth_colormap, depth_frame)
        
        cv2.rectangle(color_frame, (260, 0), (380, 480), (0, 0, 255), 5)
        cv2.rectangle(depth_colormap, (260, 0), (380, 480), (0, 0, 255), 5)
        
        cv2.imshow('color_frame', color_frame)
        cv2.imshow('depth_colormap', depth_colormap)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break      
finally:
    pipeline.stop()

print('---------------------')
haniumDB = pymysql.connect(host=HOST, 
                     user=USER, 
                     password=PASSWARD)
cursor = haniumDB.cursor()

cursor.execute('use hanium')

file = open('./cloud/container_layer.txt', 'w')
for lb in range(9):
    layer = 0
    data = np.median(np.array(depth_array[lb]))

    if lb >= 5:
        data = data - 15
    if data < 485:
        layer = 3
    elif data < 525 and data >= 485:
        layer = 2
    elif data < 555 and data >= 525:
        layer = 1
    else:
        layer = 0
    file_write = str(lb) + ' ' + str(layer) + '\n'
    cursor.execute(sql, (lb, layer))
    haniumDB.commit()
    haniumDB.close()
    
    print(file_write)
    file.write(file_write)
file.close()
        
    