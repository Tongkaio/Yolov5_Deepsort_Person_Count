import os.path

import numpy as np

import tracker
from detector import Detector
import cv2

if __name__ == '__main__':
    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    video_name = '2.mp4'
    capture = cv2.VideoCapture(os.path.join('./video', video_name))  # 这里填入视频的本地位置

    # 保存视频用的参数
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_output_name = video_name.split('.')[0] + "_output.avi"
    writer = cv2.VideoWriter(video_output_name, fourcc, fps, (960, 540), True)

    # output_log
    log_output_name = video_name.split('.')[0] + "_log.txt"
    file_path = log_output_name

    # 记录帧数
    frame = 0
    while True:
        # 读取每帧图片
        _, im = capture.read()  # 当视频帧读取完毕时，read() 方法会返回 (False, None)。
        if im is None:
            break
        frame += 1  # 更新帧数
        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)  # 返回 bounding boxes

        # 如果画面中有 bbox
        if len(bboxes) > 0:
            # list_bbox的每个元素是(x1, y1, x2, y2, label, track_id)
            # 包含四角坐标，类别标签，跟踪ID
            list_bboxs = tracker.update(bboxes, im)  # tracker 是 deepsort

            # 画框（框+类别+ID）

            output_image_frame = tracker.draw_bboxes_forguowang(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 如果画面中 没有bbox 输出原始图像
            output_image_frame = im
        pass

        if len(list_bboxs) > 0:
            content = 'frame-{} '.format(frame)
            for item_bbox in list_bboxs:  # 遍历每个 bbox，
                x1, y1, x2, y2, label, track_id = item_bbox
                check_point_x = int(x1 + ((x2 - x1) * 0.5))
                check_point_y = y2
                content += 'ID-{} {} {} '.format(track_id, check_point_x, check_point_y)
            content += '\n'
            with open(file_path, 'a') as file:
                file.write(content)

            # 清空list
            list_bboxs.clear()

            pass
        else:
            pass
        cv2.imshow('demo', output_image_frame)
        writer.write(output_image_frame)  # 保存视频
        cv2.waitKey(1)

        pass
    pass
    writer.release()  # 保存视频
    capture.release()
    cv2.destroyAllWindows()
