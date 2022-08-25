from mtcnn_pytorch.src import detect_faces, show_bboxes
from PIL import Image
import time

# load_ext autoreload
# autoreload 2
start = time.time()

# img = Image.open('mtcnn_pytorch/images/teamCV.PNG')
img = Image.open('mtcnn_pytorch/images/office1.jpg')
bounding_boxes, landmarks = detect_faces(img)
image = show_bboxes(img, bounding_boxes, landmarks)
image.show()
end = time.time()

print(end-start)

# img = Image.open('mtcnn_pytorch/teamCV.PNG')
# bounding_boxes, landmarks = detect_faces(img)
# show_bboxes(img, bounding_boxes, landmarks)

# boxe, landmarks = detect_faces(PATH)
# show_bboxes(PATH, boxe, landmarks)
