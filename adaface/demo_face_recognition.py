from __future__ import print_function, division

import torch
# import time
import numpy as np
# import cv2
import tensorrt as trt

import pycuda.driver as cuda
# import pycuda.autoinit

# import tensorrt as trt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from PIL import Image, ImageFont, ImageDraw, ImageFilter
import gradio as gr


# load model TensorRT gpu
from onnx_helper import ONNXClassifierWrapper
torch.cuda.init()
device = cuda.Device(0)
ctx = device.make_context()
PRECISION = np.float32
model_tensorrt = ONNXClassifierWrapper("finetune_ir101_gpu.trt", [1, 512], target_dtype = PRECISION)


# load model Face Detection (SCRFD - Lib: Insightface)
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


# image = Image.open(PATH)
names = np.load("name_data_dktd_align_blur.npy", allow_pickle = False)
features = np.load("features_data_dktd_align_blur.npy", allow_pickle = False)
label_numeric = np.load("labels_data_dktd_align_blur.npy", allow_pickle = False)


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    np_img = brg_img.transpose(2, 0, 1).astype(np.float32)
    return np_img


def detect(image_input, radius_blur, crop_size=112):
    image = image_input.filter(ImageFilter.GaussianBlur(radius_blur))
    image_cv2 = np.array(image)
    image_cv2 = image_cv2[:, :, ::-1].copy()   
    faces = app.get(image_cv2)
    image_cv2 = image_cv2.transpose(2, 0, 1).astype(np.float32)
    boxes = []
    faces_crop = []
    for i in range(len(faces)):
        boxes.append(faces[i].bbox)
        face = image.crop(faces[i].bbox).resize((112, 112))
        faces_crop.append(face)
    return image, faces_crop, boxes


def recognition(image_input, THRESHOLD, radius_blur):
    size = 1280
    g = (size / max(image_input.size))  # gain
    image_input = image_input.resize((int(x * g) for x in image_input.size), Image.Resampling.LANCZOS)  # resize
    image, faces_crop, boxes = detect(image_input, radius_blur)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='arial.ttf',size=np.floor(1.2e-2 * image_input.size[1] + 0.5).astype('int32'))
    for i in range(len(faces_crop)):
        inputs = to_input(faces_crop[i])
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        ctx.push()
        feature = model_tensorrt.predict(inputs)
        ctx.pop()
        score = feature @ features.T
        if np.max(score)>THRESHOLD:
            id = np.argmax(score)
            id = label_numeric[id]
            text = names[id]
        else: 
            text = 'Unknow'

        text_width, text_height = draw.textsize(text, font=font)
        draw.text((boxes[i][0]+5, boxes[i][3] - text_height-5), text, fill="yellow", font=font)

    for b in boxes:
            draw.rectangle([
                (b[0], b[1]), (b[2], b[3])
            ], outline='green', width = 3)
    return image

image_input = Image.open("anhtest.jpg")
a = recognition(image_input, 0.2, 1)


# gr.Interface(fn=recognition,
#              inputs = [gr.inputs.Image(type='pil', label="Original Image"), 
#                        gr.inputs.Number(label="Threshold (-1,1)"), 
#                        gr.inputs.Slider(0, 10, label="Radius Blur")], 
#              outputs = [gr.outputs.Image(type="pil", label="Output Image")], 
#              live=True).launch(server_name='172.26.33.18', server_port=8029)

gr.Interface(fn=recognition,
             inputs = [gr.inputs.Image(type='pil', label="Original Image"), 
                       gr.inputs.Number(label="Threshold (-1,1)"), 
                       gr.inputs.Slider(0, 10, label="Radius Blur")], 
             outputs = [gr.outputs.Image(type="pil", label="Output Image")]).launch(server_name='172.26.33.18', server_port=8029)