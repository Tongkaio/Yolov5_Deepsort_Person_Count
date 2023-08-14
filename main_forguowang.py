import numpy as np

import tracker
from detector import Detector
import cv2

offset = 0.2
if __name__ == '__main__':
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)  # 1080行，1920列，数据类型np.uint8

    # 初始化2个撞线polygon
    # 填充第一个polygon
    list_pts_blue = [[0, 500], [1920, 500], [1920, 540], [0, 540]]  # 列表类型
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)  # 转换为numpy
    # cv2.fillPoly(image, [多边形顶点array1, 多边形顶点array2, … ], RGB color)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)  # shape 1080*1920
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]  # shape 1080*1920*1

    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[0, 620], [1920, 620], [1920, 660], [0, 660]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用, color=1的是第一个polygon，=2的是第二个polygon
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX  # 设置绘制文本的字体风格
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))  # 绘制文本的位置

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./video/NVR-1.mp4')  # 这里填入视频的本地位置
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')

    # 保存视频用的参数
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, fps, (960, 540), True)

    while True:
        # 读取每帧图片
        _, im = capture.read()  # 当视频帧读取完毕时，read() 方法会返回 (False, None)。
        if im is None:
            break

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
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None, offset=offset)
            pass
        else:
            # 如果画面中 没有bbox 输出原始图像
            output_image_frame = im
        pass

        # 输出图片（加上两条撞线）cv2.add保证数据不会溢出
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:  # 遍历每个 bbox，
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * offset))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:  # 如果点在蓝色撞线区域内（上方）
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)  # 添加此点对应的行人ID到蓝线数组
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1

                        print(f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')

                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1

                        print(f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            # 蓝线和黄线里有此 ID，且当前帧已经没有此 ID，则删除从蓝线和黄线里删除此 ID
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon  # 列表拼接
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass
        # 显示文本
        text_draw = 'DOWN: ' + str(down_count) + \
                    ' , UP: ' + str(up_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(0, 0, 255), thickness=2)

        cv2.imshow('demo', output_image_frame)
        writer.write(output_image_frame)  # 保存视频
        cv2.waitKey(1)

        pass
    pass
    writer.release()  # 保存视频
    capture.release()
    cv2.destroyAllWindows()
