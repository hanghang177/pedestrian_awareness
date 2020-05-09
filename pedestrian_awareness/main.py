# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import torchvision.models as models
import torch
import traceback
import matplotlib

# https://stats.stackexchange.com/questions/265266/adjusting-s-curves-sigmoid-functions-with-hyperparameters
def sigmoid(x, ymin, ymax, L50, U50):
    a = (L50 + U50)/2
    b = 2 / abs(L50 - U50)
    c = ymin
    d = ymax - c
    sig = c + (d/(1+math.exp(b*(a-x))))
    return sig

def awareness_model(head_orientation, p_cellphone):

    if not head_orientation:
        p_looking_away = 1.0
    else:
        p_looking_away = sigmoid(head_orientation, 0.0, 1.0, 30, 60)

    p_cellphone = sigmoid(p_cellphone, 0.0, 1.0, 0.5, 0.8)

    w_looking_away = 0.6
    w_cellphone = 0.4

    p_distracted = w_looking_away * p_looking_away + w_cellphone * p_cellphone

    p_distracted /= w_looking_away + w_cellphone

    p_aware = 1 - p_distracted

    return p_aware

def awareness_color(p_aware):
    no_awareness_color = np.array([0.0, 1.0, 1.0])
    full_awareness_color = np.array([0.33, 1.0, 1.0])
    awareness_color = full_awareness_color * p_aware + no_awareness_color * (1 - p_aware)

    awareness_color_rgb = matplotlib.colors.hsv_to_rgb(awareness_color) * 255

    awareness_color_bgr = [awareness_color_rgb[2], awareness_color_rgb[1], awareness_color_rgb[0]]

    # awareness_color = full_awareness_color * awareness / 255 + no_awareness_color * (255 - awareness) / 255
    # print(awareness_color)
    return awareness_color_bgr

def get_hand_boundingbox(elbow, wrist, head_size):

    alpha = 1.5
    beta = 6

    hand_center = elbow + (wrist - elbow) * alpha

    hand_size = np.sqrt(beta * head_size)

    hand_bbox_1 = hand_center - hand_size * 0.5

    hand_bbox_2 = hand_center + hand_size * 0.5

    return hand_bbox_1.astype(int), hand_bbox_2.astype(int)

def check_valid(pt):
    return not np.array_equal(pt, np.array([0, 0]))

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('openpose_python');
        sys.path.append('yolov3')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        import pyopenpose as op
        from models import *  # set ONNX_EXPORT in models.py
        from utils.datasets import *
        from utils.utils import *
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    parser.add_argument("--output_video", default=False, help="Output to a video")
    parser.add_argument("--source", default=0, help="Video Source")
    parser.add_argument("--photo", default=False, help="Is this a photo?")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "models/"
    params["face"] = True
    params["net_resolution"] = "256x256"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()

    source = args[0].source
    output_video = args[0].output_video
    is_photo = args[0].photo

    if is_photo:
        img_source = source
        source = 0

    if source == "0":
        source = 0

    # Open capture on cv2
    cap = cv2.VideoCapture(source)
    if source == 0:
        width = 1280
        height = 720
        fps = 30
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height);
    else:
        # https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = 60
    #cap = cv2.VideoCapture(0)
    # Open writer on cv2
    if output_video:
        writer = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), fps, (width, height))

    # YOLOV3
    cfg = 'yolov3/cfg/yolov3-spp.cfg'
    names = 'yolov3/data/coco.names'
    weights = 'yolov3/weights/yolov3-spp-ultralytics.pt'
    img_size = 320
    conf_thres = 0.5
    iou_thres = 0.6
    half = False
    device = ''
    agnostic_nms = False
    augment = False
    names = load_classes(names)

    # Initialize
    device = torch_utils.select_device(device='gpu')

    # Initialize model
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    while True:
        datum = op.Datum()

        if is_photo:
            frame = cv2.imread(img_source)
        else:
            ret, frame = cap.read()

            if (not ret) and (source != 0):
                break
        im_size = frame.shape
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        #print("Face keypoints: \n" + str(datum.faceKeypoints))

        angle = None

        if datum.faceKeypoints is not None:

            nose_tip = datum.faceKeypoints[0,30,0:2]
            chin = datum.faceKeypoints[0,8,0:2]
            left_eye = datum.faceKeypoints[0,45,0:2]
            right_eye = datum.faceKeypoints[0,36,0:2]
            left_mouth = datum.faceKeypoints[0,54,0:2]
            right_mouth = datum.faceKeypoints[0,48,0:2]

            head_x_min = np.min(datum.faceKeypoints[0,:,0])
            head_y_min = np.min(datum.faceKeypoints[0,:,1])
            head_x_max = np.max(datum.faceKeypoints[0,:,0])
            head_y_max = np.max(datum.faceKeypoints[0,:,1])

            head_size = (head_x_max - head_x_min) * (head_y_max - head_y_min)

            image_points = np.array([nose_tip, chin, left_eye, right_eye, left_mouth, right_mouth], dtype = np.float32)

            print(image_points)

            # https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
            model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                            ], dtype = np.float32)

            focal_length = im_size[1]
            center = (im_size[1]/2, im_size[0]/2)
            camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = np.float32
                             )

            dist_coeffs = np.zeros((4,1), dtype = np.float32) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

            transformation_matrix_rot_only = np.array([
                [rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], 0],
                [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], 0],
                [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], 0],
                [0, 0, 0, 1]
            ])
            unit_z = np.array([[0], [0], [1], [1]])
            rot_z = np.transpose(np.matmul(transformation_matrix_rot_only, unit_z))[0][0:3]

            angle = np.degrees(np.arccos(np.dot(rot_z[0:3], -translation_vector) / (np.linalg.norm(rot_z) * np.linalg.norm(translation_vector))))

            print("Degrees: "+ str(angle))

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            draw_frame = datum.cvOutputData

        #
        #
        # Hand detection

        left_elbow = datum.poseKeypoints[0, 6, 0:2]
        left_wrist = datum.poseKeypoints[0, 7, 0:2]

        left_hand_valid = False

        if check_valid(left_elbow) and check_valid(left_wrist):
            left_bbox1, left_bbox2 = get_hand_boundingbox(left_elbow, left_wrist, head_size)
            left_hand_valid = True
        else:
            left_bbox1, left_bbox2 = -1, -1

        right_elbow = datum.poseKeypoints[0, 3, 0:2]
        right_wrist = datum.poseKeypoints[0, 4, 0:2]

        right_hand_valid = False

        if check_valid(right_elbow) and check_valid(right_wrist):
            right_bbox1, right_bbox2 = get_hand_boundingbox(right_elbow, right_wrist, head_size)
            right_hand_valid = True
        else:
            right_bbox1, right_bbox2 = -1, -1

        # Hand-held object detection

        max_x = frame.shape[0]
        max_y = frame.shape[1]

        min_x = 0
        min_y = 0

        have_cellphone_left = False
        have_cellphone_right = False
        p_cellphone_left = 0
        p_cellphone_right = 0

        if left_hand_valid:

            if left_bbox2[1] >= max_x:
                left_bbox2[1] = max_x - 1
            if left_bbox1[1] < min_x:
                left_bbox1[1] = min_x

            if left_bbox2[0] >= max_y:
                left_bbox2[0] = max_y - 1
            if left_bbox1[0] < min_y:
                left_bbox1[0] = min_y

            left_img = frame[left_bbox1[1]:left_bbox2[1] + 1, left_bbox1[0]:left_bbox2[0] + 1]

            if left_img.shape[0] == 0 or left_img.shape[1] == 0:
                pass
            else:

                img = letterbox(left_img, new_shape=img_size)[0]

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
                        #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                        for *xyxy, conf, cls in det:
                            if names[int(cls)] == 'cell phone':
                                p_cellphone_left = conf.cpu().detach().numpy()
                                have_cellphone_left = True
                                break

                    if have_cellphone_left:
                        break

        if right_hand_valid:

            if right_bbox2[1] >= max_x:
                right_bbox2[1] = max_x - 1
            if right_bbox1[1] < min_x:
                right_bbox1[1] = min_x

            if right_bbox2[0] >= max_y:
                right_bbox2[0] = max_y - 1
            if right_bbox1[0] < min_y:
                right_bbox1[0] = min_y

            right_img = frame[right_bbox1[1]:right_bbox2[1] + 1, right_bbox1[0]:right_bbox2[0] + 1]

            if right_img.shape[0] == 0 or right_img.shape[1] == 0:
                pass
            else:
                img = letterbox(right_img, new_shape=img_size)[0]

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
                        # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                        for *xyxy, conf, cls in det:
                            if names[int(cls)] == 'cell phone':
                                p_cellphone_right = conf.cpu().detach().numpy()
                                have_cellphone_right = True
                                break

                    if have_cellphone_right:
                        break

        p_cellphone = max(p_cellphone_left, p_cellphone_right)

        if p_cellphone > 0:
            print("Cellphone")
            print(p_cellphone)

        if left_hand_valid:
            if have_cellphone_left:
                text = "Phone"
                color = (0, 0, 255)
            else:
                text = "No Phone"
                color = (0, 255, 0)
            cv2.putText(draw_frame, text, tuple(left_bbox1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(draw_frame, tuple(left_bbox1), tuple(left_bbox2), color, 2)
        if right_hand_valid:
            if have_cellphone_right:
                text = "Phone"
                color = (0, 0, 255)
            else:
                text = "No Phone"
                color = (0, 255, 0)
            cv2.putText(draw_frame, text, tuple(right_bbox1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(draw_frame, tuple(right_bbox1), tuple(right_bbox2), color, 2)

        p_aware = awareness_model(angle, p_cellphone)
        color = awareness_color(p_aware)

        cv2.line(draw_frame, p1, p2, (255,0,0), 2)
        cv2.circle(draw_frame, p1, 20, color, 4)
        text = "P_aware = {0:.2f}".format(p_aware)
        cv2.putText(draw_frame, text, (head_x_min, head_y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        if is_photo:
            cv2.imwrite("test.jpg", draw_frame)
            break

        if output_video:
            writer.write(draw_frame)
        cv2.imshow("Data", draw_frame)
        cv2.waitKey(1)

    cap.release()
    writer.release()
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    cap.release()
    writer.release()
    print(e)
    print(traceback.format_exc())
    sys.exit(-1)
