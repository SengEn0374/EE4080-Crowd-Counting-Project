'''
predict a given video
'''

from models.vgg import vgg19
import torch
import numpy as np
import torchvision.transforms as standard_transforms
from PIL import Image
import os
import cv2
import time

# initialize model
PATH = 'C:/Users/crono/Desktop/generalizedloss_weights/ucf_vgg19_ot_84.pth'
model = vgg19().eval().cuda()
# model = CSRNet()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)

# settings
asp_ratio = [(1024,576), (1152,648), (1280,720), (1366,768), (1600,900), (1920,1080), (2560,1440), (3840,2160), (640,640)]
mean_std_nwpu = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
mean_std_imgnet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std_imgnet)])

# max density map value = 22.035421 from SHT_A_trng
# density_norm_factor = 22.035421  # not used
max = 0.0

# cap = cv2.VideoCapture("../00007.MTS") # test video
cap = cv2.VideoCapture(0) # webcam test
# cap = cv2.VideoCapture("http://192.168.0.192:81/stream")  # web cam server

while True:
    ret, _frame = cap.read()
    frame = cv2.resize(_frame, asp_ratio[0]) # set to smallest for speed. trade off some accuracy
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)  # convert to PIL Image. as torchvision only works with pil images
    start = time.time()

    # img = transform(Image.open(os.path.join(imgs_path, image)).convert('RGB')).cuda()
    img = transform(image.convert('RGB')).cuda()
    # img = transform(image.convert('RGB'))
    # img = transform(image).cuda()

    with torch.no_grad():
        output = model(img.unsqueeze(0))

    count = int(output.detach().cpu().sum().numpy())
    print("Predicted Count : ", count)
    print("%.3f" % (time.time() - start))

    # fetch density map
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
    # print(temp.min()) # negative values causing wrap around
    temp = temp - temp.min()  # shift to zero base value
    if temp.max() > max:
        max = temp.max()
        print("max", max)
    temp = cv2.resize(temp, (frame.shape[1], frame.shape[0]))

    # scale back to img size + convert to unint8 for map reading as image
    scale = 255 / temp.max()  # want to set fixed color mapping instead?
    # scale = 255/ density_norm_factor

    temp = (temp * scale).astype(np.uint8)
    temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)

    # convert frame back to rgb (cv2 read as bgr)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # overlay frame with density map
    overlay = cv2.addWeighted(frame, 0.5, temp, 0.5, 0.0)

    # add count number on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, str(count), (50, 50), font, 1, (225, 225, 225), 3)

    cv2.imshow('overlay', overlay)
    # cv2.imshow('temp', temp)  # density map only
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
    break  # for testing purpose
cap.release()
cv2.destroyAllWindows()
