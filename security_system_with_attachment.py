# Import libraries
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

# Email settings
smtp_port = 587
smtp_server = "smtp.gmail.com"
from_email = "sskw5651@gmail.com"
email_list = ["sskw5651@gmail.com", "werhanikamel51@gmail.com"]
password = "pvdlzkpximyaijhm" # Placeholder, should be securely referenced
subject = "Security Alert!!"

def send_emails_async(email_list, from_email, object_detected, image_data):
    threading.Thread(target=send_emails, args=(email_list, from_email, object_detected, image_data)).start()

def send_emails(email_list, from_email, object_detected, image_data):
    message_body = f'ALERT - {object_detected} person(s) detected!!'

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = ", ".join(email_list)
    msg['Subject'] = subject
    msg.attach(MIMEText(message_body, 'plain'))

    attachment_package = MIMEBase('application', 'octet-stream')
    attachment_package.set_payload(image_data)
    encoders.encode_base64(attachment_package)
    attachment_package.add_header('Content-Disposition', "attachment; filename=alert.png")
    msg.attach(attachment_package)

    text = msg.as_string()

    with smtplib.SMTP(smtp_server, smtp_port) as TIE_server:
        TIE_server.starttls()
        TIE_server.login(from_email, password)
        TIE_server.sendmail(from_email, email_list, text)

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False
        self.model = YOLO("weights/yolov8l.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        results = self.model(im0)
        return results

    def crop_image(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        xyxy = np.round(xyxy).astype(int)
        x1, y1, x2, y2 = xyxy
        cropped_img = image[y1:y2, x1:x2]
        return cropped_img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
        zone_polygon = (ZONE_POLYGON * np.array([1920, 1080])).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple([1920, 1080]))
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red(), thickness=2, text_thickness=4, text_scale=2)

        while True:
            start_time = time()
            ret, im0 = cap.read()
            if not ret:
                break

            results = self.predict(im0)[0]
            detections = sv.Detections.from_yolov8(results)
            class_0 = detections[detections.class_id == 0]
            
            labels = [f"{self.model.model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in class_0]
            im0 = box_annotator.annotate(scene=im0, detections=class_0, labels=labels)
            
            zone.trigger(detections=class_0)
            person_count = len(zone.trigger(detections=class_0))
            print("Number of persons:", person_count)
            
            if class_0:
                if not self.email_sent:
                    try:
                        cropped_image = self.crop_image(im0, class_0.xyxy[0])
                        _, image_data = cv2.imencode('.png', cropped_image)
                        send_emails_async(email_list, from_email, person_count, image_data.tobytes())
                        self.email_sent = True
                    except IndexError:
                        print("IndexError: Unable to access element at index 0. class_0.xyxy may be empty.")
            else:
                self.email_sent = False
            
            im0 = zone_annotator.annotate(scene=im0)
        
            cv2.imshow("Security_System", im0)
            if cv2.waitKey(30) == 27:
                break

detector = ObjectDetection(capture_index=0)
detector()
