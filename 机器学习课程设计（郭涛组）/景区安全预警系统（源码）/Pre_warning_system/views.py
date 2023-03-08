import os.path
import cv2
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render, HttpResponse, redirect
from django.views.decorators.csrf import csrf_exempt
from Pre_warning_system.Scenic_spot_safety_waring_yolov5.detect import YOLO
import sys

import argparse
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from Pre_warning_system.Scenic_spot_safety_waring_yolov5.models.experimental import attempt_load
from Pre_warning_system.Scenic_spot_safety_waring_yolov5.utils.datasets import LoadStreams, LoadImages
from Pre_warning_system.Scenic_spot_safety_waring_yolov5.utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from Pre_warning_system.Scenic_spot_safety_waring_yolov5.utils.plots import plot_one_box
from Pre_warning_system.Scenic_spot_safety_waring_yolov5.utils.torch_utils import select_device, load_classifier, \
    time_synchronized

sys.path.insert(0, 'Pre_warning_system/Scenic_spot_safety_waring_yolov5')

res_dir = ''  # 结果位置
all_per = 0  # 人数统计


# Create your views here.
@csrf_exempt
def index(request):
    """
    预警系统首页
    :param request:
    :return:
    """
    if request.method == "GET":

        # target_dir = "jiaoshi_02.jpg"
        # res_dir = dec.main(target_dir)
        # res_dir += "/" + target_dir
        # print(res_dir)
        global all_per
        all_per = 0  # 页面刷新，人数清零
        return render(request, 'index.html')
    else:
        upload_file = request.FILES.get("upload_file")
        if upload_file is not None:
            file_name = upload_file.name
            print("文件名称：" + file_name)
            file_path = os.path.join("runs", "target_dirs", file_name)
            print("前端文件保存路径：" + file_path)
            # 把输入的文件写到指定位置
            with open(file_path, 'wb') as f:
                for chunk in upload_file.chunks():
                    f.write(chunk)

            # 对输入文件进行处理
            yolo = YOLO()
            global res_dir
            res_dir = yolo.detect(file_path)
            # 将处理结果路径给前端界面
            all_per = f"{yolo.all_per}"
            # 人数统计返回前端页面
            return redirect("/result/")
        else:
            print("False!")


def result(request):
    """
    结果返回页面
    :param request:
    :return:
    """
    global res_dir
    return render(request, 'result.html', {'res_dir': res_dir, 'all_per': all_per})


# 对视频流数据进行处理
class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.all_per = 0
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        save_path = "runs/temp.jpg"
        cv2.imwrite(save_path, image)
        # 在这里处理视频帧
        #  基本参数配置
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='Pre_warning_system/Scenic_spot_safety_waring_yolov5/weights/best.pt',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=save_path, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results', default=True)
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args(args=[])
        print(opt)
        check_requirements(exclude=('pycocotools', 'thop'))

        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Stream results
                if view_img:
                    self.all_per = 0  # 总人数进行统计
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        self.all_per += n
                    cv2.putText(im0, f"persons: {self.all_per} ", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),
                                2)
                    print(f"人数: {self.all_per} ")  # 输出总人数
                    global all_per
                    all_per = f"{self.all_per}"
                    # cv2.imshow(str(p), im0)
                    # cv2.waitKey(30)  # 30 millisecond

                    # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
                    ret, jpeg = cv2.imencode('.jpg', im0)

        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video(request):
    """
    视频流路由。将其放入img标记的src属性中。
    例如：<img src='https://ip:port/uri' >
    """
    vd = VideoCamera()  # 实例化摄像类
    return StreamingHttpResponse(gen(vd), content_type='multipart/x-mixed-replace; boundary=frame')


def ger_per(request):  # 获取摄像时总人数
    return JsonResponse({"all_per": all_per})
