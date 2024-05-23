"""
    This file is a TensorRT version modified from `detector.py`.
"""
import cv2
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
# from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger()
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4

# load coco labels
# categories = ["person", "head", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#               "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#               "hair drier", "toothbrush"]

categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
              "hair drier", "toothbrush"]

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # boxes[:, 0].clamp_(0, img_shape[1])  # x1
    # boxes[:, 1].clamp_(0, img_shape[0])  # y1
    # boxes[:, 2].clamp_(0, img_shape[1])  # x2
    # boxes[:, 3].clamp_(0, img_shape[0])  # y2
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 0], 0, img_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 0], 0, img_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 0], 0, img_shape[0])

class Detector:

    def __init__(self, engine_file_path):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

        # Remove the context pushed by `cuda.Device(0).make_context()`
        self.cfx.pop()

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path, convert it to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image_raw, image, h, w


    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y = np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        # back to x,y,w,h format
        y[:, 2] = y[:, 2] - y[:, 0]
        y[:, 3] = y[:, 3] - y[:, 1]
        return y


    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si].tolist()
        classid = classid[si].tolist()
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONF_THRESH, nms_threshold=IOU_THRESHOLD)
        result_boxes = np.array(boxes)
        result_scores = np.array(scores)
        result_classid = np.array(classid)
        if len(boxes) > 0:
            result_boxes = result_boxes[indices, :]
            result_scores = result_scores[indices]
            result_classid = result_classid[indices]
        # back to x1, y1, x2, y2 format
        result_boxes = np.reshape(result_boxes, (-1, 4))
        result_boxes[:, 2] = result_boxes[:, 2] + result_boxes[:, 0]
        result_boxes[:, 3] = result_boxes[:, 3] + result_boxes[:, 1]
        #
        results_trt = []
        for i in range(len(result_boxes)):
            cid = result_classid[i]
            label = categories[int(cid)]
            if label not in ['head']:  # only show categorie:'head', (other choices: 'person')
                continue

            x1, y1 = int(result_boxes[i][0]), int(result_boxes[i][1])
            x2, y2 = int(result_boxes[i][2]), int(result_boxes[i][3])
            # conf = result_scores[i][0]
            conf = result_scores[i]
            results_trt.append(
                (x1, y1, x2, y2, label, conf))

        return results_trt


    def detect(self, im):
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        # input_image, image_raw, origin_h, origin_w = self.preprocess_image(im)  # 2024/5/22这里似乎输出顺序存在错误
        image_raw, input_image, origin_h, origin_w = self.preprocess_image(im)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        trt_outputs = host_outputs[0]

        # Do postprocess
        results_trt = self.post_process(trt_outputs, origin_h, origin_w)
        return results_trt

if __name__ == '__main__':
    import ctypes
    PLUGIN_LIBRARY = "weights/libyolov5splugins.so"
    engine_file_path = "weights/yolov5s.engine"
    img_path = "video/first_frame.jpg"

    ctypes.CDLL(PLUGIN_LIBRARY)
    detector = Detector(engine_file_path)
    img = cv2.imread(img_path)

    img = cv2.resize(img, (800, 512))
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
