import net
import torch
# from face_alignment import align
from src import detect_faces, show_bboxes
from PIL import Image
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt


# Parameter:
PATH = r"Test_Find_Class\anhvi.PNG"

THRESHOLD = 0.3


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


def similarity_scores(feature_1, feature_2):
    similarity_scores = feature_1 @ feature_2.T
    return similarity_scores


names = []
for name in os.listdir("data_VC_khongkhautrang"):
    names.append(name)

name_index = {name:index for index,name in enumerate(names, start=1)}
index_name = {index:name for index,name in enumerate(names, start=1)}

label = []
data = []
label_numeric = []

def find_name(label):
    start = label.index('-') + 2
    end = start + label[start:].index('-') - 1
    return label[start:end]

NAME =[]

for name in os.listdir("data_VC_khongkhautrang"):
    NAME.append(find_name(name))
    for path in os.listdir("data_VC_khongkhautrang/"+name):
        label.append(find_name(name))
        data.append(os.path.join("data_VC_khongkhautrang", name, path))
        label_numeric.append((name_index[name]))


label = np.array(label)
data = np.array(data)
label_numeric = np.array(label_numeric)

adaface_models = {
    'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
}

architecture = 'ir_50'
assert architecture in adaface_models.keys()
model = net.build_model(architecture)
statedict = torch.load(adaface_models[architecture])['state_dict']
model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
model.load_state_dict(model_statedict)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

boxe, landmarks = detect_faces(PATH)
show_bboxes(PATH, boxe, landmarks)

# boxe, aligned_rgb_img = align.get_aligned_face(PATH)
boxes = []

for i in range(len(boxe)):
    boxes.append(boxe[i][0:4].tolist())

print(boxes)

features = []
scores = []


def find_name(feature):
    for i in range(len(feature_averages)):
        score = similarity_scores(feature_averages[i], feature)
        scores.append(score)
    if np.max(scores) > THRESHOLD:
        ID = np.argmax(scores)
        text = NAME[ID]
    else:
        text = "unknow"
    return text


# creating a image object
image = Image.open(PATH)
draw = ImageDraw.Draw(image)

for i in range(len(aligned_rgb_img)):
    input = to_input(aligned_rgb_img[i])
    feature, _ = model(input)
    feature = np.array(feature)
    # features.append(feature)

    feature_averages = np.load("features_average_class_data_VC.npy")
    scores = []

    text = find_name(feature)

    # for i in range(len(feature_averages)):
    #     score = similarity_scores(feature_averages[i], feature)
    #     scores.append(score)
    #
    # # if np.max(scores) > THRESHOLD:
    # #     id = np.argmax(scores) + 1
    # # else:
    # #     id = "unknow"
    # id = np.argmax(scores)
    # text = NAME[id]


    # image = Image.open(PATH)
    # draw.rectangle(boxes[i], outline="blue")
    # # specified font size
    font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 14)
    # # drawing text size
    # draw.text((5, 5),  text, fill='red', font=font, align="center")
    # image.show()
    print(text)

    left = boxes[i][0]+5
    top = boxes[i][1]+5
    right = boxes[i][2]+5
    bottom = boxes[i][3]+5

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(text, font=font)
    print(text_width)
    print(text_height)
    draw.rectangle(((left, bottom - text_height), (left+text_width+10, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left+5, bottom - text_height), text, fill="white", font=font)

del draw

image.show()