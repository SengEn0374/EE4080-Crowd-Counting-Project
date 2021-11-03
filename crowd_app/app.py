import numpy as np
import argparse
import time
from datetime import datetime
from flask import Flask, render_template, Response
import requests
import cv2
from PIL import Image
import torch
import torchvision.transforms as standard_transforms
from models.vgg19csr import vgg19


app = Flask(__name__)


def find_camera(list_id):
    return cameras[int(list_id)]


def gen_frames(camera_id):
    cam = find_camera(camera_id) 
    cap = cv2.VideoCapture(cam)
    
    # init alert funct vars for Telegram API
    ignore_event = False        # init flag. Ignores alert if overcrowd happen within set interval
    event_time = 0              # init elapsed time to ensure first event is caught
    interval = args.interval    # num seconds between each notifications from same camera

    while True:
        start = time.time()
        success, frame = cap.read()  # read the camera frame
        time_stamp = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        # print(frame.shape)
        if not success:
            print('failed to retrieve frame, please restart programme')
            break
        output, count = crowd_counter(frame)
        if args.density_map:
            frame = add_density_map(output, frame)

        # add count number on video
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = put_count(frame, time_stamp, count, font)

        # check threshold and add red border if overcrowding, send alert
        if (camera_id=='0' and count>thresh_0) or (camera_id=='1' and count>thresh_1) or (camera_id=='2' and count>thresh_2):
            frame = cv2.rectangle(frame, (0,0), (frame.shape[1]-2, frame.shape[0]-2), (0,0,255), 5)
            cv2.putText(frame, "[!]", (10, frame.shape[0]-20), font, 1, (0,0,225), 2)
            time_elapsed = time.time()-event_time
            # print(time_elapsed)
            if ignore_event and time_elapsed>interval:
                ignore_event = False
            if not ignore_event:
                event_time = time.time()    # start timer
                if args.telegram_alert:
                    send_msg(camera_id)
                    print(f'alert sent area {camera_id}')
                else:
                    print(f'overcrowding in area {camera_id}')
                ignore_event = True

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        print(f'\rfps {1/(time.time()-start)}', end='')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def crowd_counter(frame):
    """
    :param frame: numpy array of an image in BGR color format
    :return: output, count: predicted density map and, predicted count
    """
    _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert back to rgb for pil image
    image = Image.fromarray(_frame)  # convert to PIL Image. torchvision only works with pil images
    img = transform(image.convert('RGB')).to(device)
    with torch.no_grad():
        output = model(img.unsqueeze(0))
    count = int(output.detach().cpu().sum().numpy())
    return output, count


def put_count(frame, time_stamp, count, font):
    cv2.putText(frame, str(time_stamp), (frame.shape[1] - 200, 20), font, 0.5, (225, 225, 225), 2)
    cv2.putText(frame, 'count: ' + str(count), (10, 30), font, 1, (225, 225, 225), 2)
    return frame


def add_density_map(output, frame):
    # fetch density map
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
    # negative values causing wrap around for some models. Offset back to min=0
    temp = temp - temp.min()

    # scale density map to img size + normalise to range 0-255 + convert to unint8 for map reading as image
    temp = cv2.resize(temp, (frame.shape[1], frame.shape[0]))
    if temp.max() < 1e-5:
        # prevent divide by 0 and flickering overlay
        scale = 0
    else:
        scale = 255 / temp.max()
    temp = (temp * scale).astype(np.uint8)
    temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.5, temp, 0.5, 0.0)
    return overlay


def send_msg(location_id):
    alert = f'Overcrowding in location {location_id} detected!'
    base_url = f'https://api.telegram.org/bot2030720373:AAFJLJkMssS1gMhMTkJoZyDdNDL8HtIrrpM/sendMessage?chat_id=-724154343&text={alert}'
    requests.get(base_url)


# test with youtube video
import pafy
def get_yt_url(yt_link):
    url = yt_link
    video = pafy.new(url)
    # best = video.getbestvideo(preftype="mp4")  # get highest quality
    # vid = best.url
    vid_ = video.streams[0] # get lowest quality just for test
    print(video.streams)
    # input('?')
    vid = vid_.url
    return vid


@app.route('/video_feed/<string:list_id>/', methods=["GET"])
def video_feed(list_id):
    return Response(gen_frames(list_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html', camera_list=len(cameras), camera=cameras)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowd Monitoring App')
    # parser.add_argument('--model', default='vgg19_ucf', type=str, help='vgg19_ucf, vgg19_nwpu, vgg19csr, csrnet, scar, sfcn')
    parser.add_argument('--weight_dir', default='weights/vgg19csr1_best_val_750ep_52.30mae_226.19mse.pth', help='model\'s weight path', )
    parser.add_argument('--device', default=0, type=int, help='assign device')
    parser.add_argument('--threshold_1', default=1, type=int, help='camera 1 threshold before alert')
    parser.add_argument('--threshold_2', default=1, type=int, help='camera 2 threshold before alert')
    parser.add_argument('--threshold_3', default=1, type=int, help='camera 3 threshold before alert')
    parser.add_argument('--interval', default=900, type=int, help='interval between each alert notification')
    parser.add_argument('--density_map', default=True, type=bool, help='indicate to display density map on monitor')
    parser.add_argument('--telegram_alert', default=True, type=bool, help='False if not using internet connection')
    args = parser.parse_args()

    # initialize model
    PATH = args.weight_dir
    device = torch.device(args.device)
    model = vgg19().eval().to(device)
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint)

    # hard settings
    # asp_ratio = [(1024, 576), (1152, 648), (1280, 720), (1366, 768), (1600, 900), (1920, 1080), (2560, 1440), (3840, 2160), (640, 640)]
    mean_std_nwpu = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    mean_std_imgnet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std_nwpu)])

    # list of default camera access
    cameras = [
        0, # testing with webcam
        # get_yt_url('https://www.youtube.com/watch?v=qcKi2LtDEl0'),    # city hall video1
        # get_yt_url('https://www.youtube.com/watch?v=xIdUcMv9COg') # city hall videos 2
        # get_yt_url('https://www.youtube.com/watch?v=E1DyTTwyPZE'), # haj crowd, very high
        'http://192.168.0.192:81/stream',
        'http://192.168.0.104:81/stream'
    ]

    # camera threshold counts
    thresh_0 = args.threshold_1
    thresh_1 = args.threshold_2
    thresh_2 = args.threshold_3

    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
