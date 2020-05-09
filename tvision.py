import cv2
import numpy

import sys

sys.path.append('yolov3')

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

try:
    cap = cv2.VideoCapture(0)

    cfg = 'yolov3/cfg/yolov3-spp.cfg'
    names = 'yolov3/data/coco.names'
    weights = 'yolov3/weights/yolov3-spp-ultralytics.pt'
    img_size = 512
    conf_thres = 0.3
    iou_thres = 0.6
    half = False
    device = ''
    agnostic_nms = False
    augment = False
    names = load_classes(names)

    # Initialize
    device = torch_utils.select_device(device)

    # Initialize model
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    while True:
        ret, frame = cap.read()

        img = letterbox(frame, new_shape=img_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False, classes = None, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

        cv2.imshow("Data", frame)
        cv2.waitKey(1)

except Exception as e:
    print(e)
    sys.exit(-1)
