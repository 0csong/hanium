import cv2
import json
import base64
import requests
import time

count = 1
timestamp = round(time.time())

image = cv2.imread("example1.png")
ret, buffer = cv2.imencode('.png', image)
png_as_text = base64.b64encode(buffer)

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

data = json.dumps(data)
response = requests.post(ocr_invoke_url, headers=headers, data=data)
res = json.loads(response.text)

res_array = res.get('images')
for list in res_array[0].get('fields'):
    print(list.get('inferText'))