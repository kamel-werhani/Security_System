
# Security_System

The Security System Project utilizing YOLOv8 integrates advanced computer vision capabilities to enhance security measures. This system detect person after send a message through email containing number of persons detected and screenshot of the detected person.

![screen01](https://github.com/user-attachments/assets/77899e8d-84b4-4171-b6cd-c08b55ae12e5)
![demo (4)](https://github.com/user-attachments/assets/e9de1b28-7a9a-4d9e-b9d5-9014b6db0ca6)


## Deployment

Install requirements
```bash
$ pip install -r requirement
```
Navigate to App Password Generator, designate an app name such as "security project," and obtain a 16-digit password. Copy this password and paste it into the designated password field as instructed.

```bash
password = ""
from_email = ""  # must match the email used to generate the password
to_email = ""  # receiver email
```
Server creation and authentication

```bash
server = smtplib.SMTP("smtp.gmail.com: 587")
server.starttls()
server.login(from_email, password)
```


## Inference demo

1. Send notification without screenshot.

```bash
$ python .\security_system.py
```


2. Send notification with screenshot
```bash
$ python .\security_system_with_attachment.py
```
