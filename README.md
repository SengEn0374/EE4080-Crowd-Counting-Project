# EE4080-Crowd-Counting-Project
A final year project, this project consists of two parts 
1. crowd counting model developement, 
2. the crowd counting webapp.

## Trained Models
The trained models can be downloaded with permission from [GoogleDrive](https://drive.google.com/drive/folders/1drinTf0G6LGF8Low9Yx0f2xX6rAbkkYB?usp=sharing).
You should respecify the path to the model when running the app.

## Requirements
- python 3.8
- pytorch 1.9.0 (py3.8_cuda11.1_cudnn8_0)
- opencv 4.0.1
- pillow 8.2.0
- flask 2.0.1
- colorama
- requests 2.25.1
- pafy
- youtube-dl
- cuda supported GPU

## Usage Example
python app.py --weight (ie ./weights/myModel.pth) 
              --threshold1 80
              --threshold2 30
              --threshold3 600
              --density_map True
              --telegram_alert True <br><br>
Note* If only using system locally, set telegram_alert to False

### Accessing live monitor video: 
http://<app's host machine ipv4 address>:5000 (local network only)
