import numpy as np
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
ap.add_argument('-r','--result',required=True,help='path to result image')
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"]) #组合路径
LABELS = open(labelsPath).read().strip().split("\n")  #读入并分隔数据

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8") #每个label有3个0-255的uint8类型值表示rgb
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])




###构建网络开始传播
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #dnn.readNetFromDarknet构建darknet的网络

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #得到输出的网络接口

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)

#图像预处理
#blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True，crop=False,ddepth = CV_32F )
#减均值mean
##同通道中减去所有像素均值
##通像素中减去所有通道均值
#图像缩放scalefactor，像素要乘以的值
#通道交换OpenCV默认是BGR，减均值顺序为RGB 设置为TRUE则避免矛盾
#size,这是神经网络，真正支持输入的值
#ddepth, 输出blob的深度，选则CV_32F or CV_8U
#cropcrop,如果crop裁剪为真，则调整输入图像的大小，使调整大小后的一侧等于相应的尺寸，另一侧等于或大于。然后，从中心进行裁剪。
# 如果“裁剪”为“假”，则直接调整大小而不进行裁剪并保留纵横比





###开始传播
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []
# loop over each of the layer outputs
###剔除置信度小的框，将置信度大的框的x,y,w,h置信度，类别记录下来
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]

        #detection中前4个数为box的x,y,w,h然后为置信度表示anchor中可能目标的概率，后面为分类器
        #detection为一个box，每个cell有5个box

        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            #将x,y,h,w缩放到原图像大小
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2)) #左上角尺寸
            y = int(centerY - (height / 2)) #左上角尺寸

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes

###
#NMS NMS NMS NMS NMS NMS
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

## cv2.dnn.NMSBoxes()
#输入第一个参数为
#第二个参数为图像的置信度
##第三第四个参数为自己设定的阈值




###表上框和信息

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

# show the output image
cv2.imwrite(args['result'],image)
cv2.imshow("Image", image)
cv2.waitKey(0)