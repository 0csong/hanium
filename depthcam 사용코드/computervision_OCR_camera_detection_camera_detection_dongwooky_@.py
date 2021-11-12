import cv2
import sys
import time
import numpy as np

import json
import base64
import requests

import boto3
import pymysql

class CameraDetection:
    
    def __init__(self):
        self.frameWidth = 640   #카메라 크기 초기화
        self.frameHeight = 480
        self.frame = np.zeros((self.frameHeight, self.frameWidth), np.uint8)    #카메라 프레임 초기화
        self.approx_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32) #꼭지점 배열 초기화
        
        self.srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)   #Perspective source 점 4개 초기화
        self.dstQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)   #Perspecrive 점 4개 초기화
        
        self.dstWidth = 0   #펼칠 프레임 크기 초기화
        self.dstHeight = 0
        self.receipt_list = []
    
    #외곽선 만들기
    def make_contours(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #(전처리) - 그레이스케일
        frame_unnoise = cv2.bilateralFilter(frame_gray, -1, 10, 5)  #(전처리) - 양방향 필터
        _, frame_binary = cv2.threshold(frame_unnoise, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)   #이진화 - OTSU알고리즘
        contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #외곽선 검출 - 외곽 검출
        return contours
    
    #사각형 찾기
    def find_rectangle(self, frame, contours):
        for points in contours:
            if cv2.contourArea(points) < 1000:  #크기가 1000이하인 contour들은 무시
                continue
            approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True) #외곽선 근사화
            if len(approx) != 4:    #사각형이 아닌것들은 무시
                continue
            self.approx_points = approx
            cv2.drawContours(frame, points, -1, (255, 0, 255), 5)   #외곽선 분홍색 굵기 5 선으로 표시
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            if cv2.contourArea(points) > self.frameWidth * self.frameHeight * 0.3: #일정 크기 이상이면 찾기 스탑
                return True         

    def reorder_points(self, points):
        idx = np.lexsort((points[:, 1], points[:, 0])) #x좌표로 먼저 정렬 후 y좌표로 정렬
        points = points[idx] #정렬
        
        if points[0, 1] > points[1, 1]:
            points[[0, 1]] = points[[1, 0]] #0번이 1번보다 아래에 있다면 교체
            
        if points[2, 1] < points[3, 1]:
            points[[2, 3]] = points[[3, 2]] #2번이 3번보다 아래에 있다면 교체
        
        self.dstWidth = round(2 * (points[3, 0] - points[0, 0]))
        self.dstHeight = round(2 * (points[1, 1] - points[0, 1]))   #결과 프레임 크기 설정
        
        self.dstQuad = np.array([[0, 0], [0, self.dstHeight], 
                                 [self.dstWidth, self.dstHeight], [self.dstWidth, 0]], np.float32) #결과 프레임 크기 설정
        
        return points
    
    def make_return_image(self, frame):
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #전처리 그레이스케일
        image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)    #전처리 가우시안 필터
        return image_blur

    def camera_detect(self):
        capture = cv2.VideoCapture(1)   #웹캠 켜기
        if not capture.isOpened():
            print('Camera open failed!')
            sys.exit()
        
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)  
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)    #웹캠마다 프레임 달라질수 있으니 프레임 고정
        
        container_ocr = []
        
        HOST = "database-hanium.cijeogglkrqn.ap-northeast-2.rds.amazonaws.com"
        USER = "admin"
        PASSWARD = "gksdldma1234!"
        
        sql = 'INSERT INTO ocr (title, date) VALUES (%s, %s)'
        
        haniumDB = pymysql.connect(host=HOST, user=USER, password=PASSWARD)
        
        cursor = haniumDB.cursor()
        cursor.execute('use hanium')
        
        while True:
            ret, self.frame = capture.read()    #웹캠에서 프레임 가져오기
            if not ret:
                print("Frame read error!")
                sys.exit()
    
            contours = self.make_contours(self.frame)   #외곽선 검출
            
            is_rectangle = self.find_rectangle(self.frame, contours)    #사각형 확인
            
            if is_rectangle == True:    #사각형이면 멈추기
                count = 1
                self.srcQuad = self.reorder_points(self.approx_points.reshape(4, 2).astype(np.float32)) #꼭지점 정렬
                perspective = cv2.getPerspectiveTransform(self.srcQuad, self.dstQuad)   #Perspective 행렬 반환
                dst_frame = cv2.warpPerspective(self.frame, perspective, (self.dstWidth, self.dstHeight))   #Perspective 변환
                timestamp = round(time.time())
                ret, buffer = cv2.imencode('.png', dst_frame)   #이미지를 png형태로 변환
                png_as_text = base64.b64encode(buffer)  #이미지를 b64로 인코딩
    
                x_ocr_secret = "ZlBWaUFqdFp4bVFZTUpnTHZucGVwZFVOZUt3cllVY0o="
                ocr_invoke_url = "https://f04b1d88f83e41a6b1df8ce399b61fb5.apigw.ntruss.com/custom/v1/9782/eed9df9be97433637c2950146e96ff0956759904a38b1e5e6a9a69a7f6691a68/general"
                uuid = "ce3e4c43-4c69-4188-ba2a-8ad24d9615e4"
    
                headers = {
                    "X-OCR-SECRET" : x_ocr_secret,
                    "Content-Type" : "application/json"
                    }
                
                data = {
                        "version" : "V1",
                        "requestId" : uuid,
                        "timestamp" : timestamp,
                        "images" : [
                            {
                                "format" : "png",
                                "data" : png_as_text.decode('utf-8'),
                                "name" : "sample_image_{}".format(count)
                                }
                            ]
                        }
    
                data = json.dumps(data) #파이썬 객체를 json로 변환
                response = requests.post(ocr_invoke_url, headers=headers, data=data)    #API로 보내기
                res = json.loads(response.text) #json을 객체로 변환
                
                res_array = res.get('images')
                container_label = open('./label/container_label.txt', 'r', encoding='utf-8')
                container_label_lines = container_label.readlines()
                
                for c_list in res_array[0].get('fields'):
                    for line in container_label_lines:
                        if (c_list.get('inferText')).find(line.split('_')[0]) > -1:
                            container_ocr.append(line.rstrip('\n'))
                container_ocr = list(set(container_ocr))
                for cont_list in container_ocr:
                    print(cont_list)
                
            cv2.imshow('Frame', self.frame) #프레임 출력
            if cv2.waitKey(33) == ord('q'): #33ms마다 프레임 바꾸기 q누르면 멈춤
                break
        for cont_list in container_ocr:
            cont_time = '2021100{}'.format(np.random.randint(1, 9))
            cont = "/".join([cont_time, cont_list])
            print(cont)
            cursor.execute(sql, (cont_list, cont_time))
        haniumDB.commit()
        haniumDB.close()
        capture.release()   #웹캠 객체 풀기
        cv2.destroyAllWindows() #모든 창 닫기

detection = CameraDetection()
detection.camera_detect()
cv2.waitKey()
cv2.destroyAllWindows()