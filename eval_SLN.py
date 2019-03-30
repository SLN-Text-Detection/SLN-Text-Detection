# coding:utf-8
'''
Evaluation for SLN.
'''
import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import zipfile

import locality_aware_nms as nms_locality
import lanms
import locality_aware_nms_v2 as nms_locality

gpu_options = tf.GPUOptions(allow_growth=True)

tf.app.flags.DEFINE_string('test_data_path', 'ch4_test_images', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'SLN_Model_IC15', '')
tf.app.flags.DEFINE_string('txt_output_dir', 'txt_SLN_Model_IC15', '')
tf.app.flags.DEFINE_string('img_output_dir', 'img_SLN_Model_IC15', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model_SLN as model
from icdar_SLN import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2048):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio    2400
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    # limit the min side
    min_side_len = 640
    if min(resize_h, resize_w) < min_side_len:
        ratio = float(min_side_len) / max(resize_h, 32) if resize_h < resize_w else float(min_side_len) / max(resize_w,
                                                                                                              32)
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, scale = 4, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param scale: based on the ratio of original image,
                  for examle: 4 denotes the size of score_map needs to be magnified 4 times to
                  reach the original size.
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*scale, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()

    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)              # Pure python
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)             # Python + C++

    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return np.array([]), timer

    # Filter some low score boxes by the average score map
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // scale, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]


    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def check_and_validate(poly):
    '''
    validate the clockwise.
    :param poly:
    :return:
    '''
    p_area = polygon_area(poly)
    if abs(p_area) < 1:
        print("invalid poly")

    if p_area > 0:
        print("poly in wrong direction")
        poly = poly[(0,3,2,1),:]
    return np.array(poly)

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.txt_output_dir)
        os.makedirs(FLAGS.img_output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry, F_score_g, F_geometry_g = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            sum_time = 0.0
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry, score_g, geometry_g = sess.run([f_score, f_geometry, F_score_g, F_geometry_g], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                # Testing for each scale output
                # boxes_list = []
                #
                # for scale in [3]: # [0,1,2,3]
                #     boxes, timer = detect(score_map=score[scale], geo_map=geometry[scale], timer=timer,
                #                           scale=2 ** (4 - scale + 1))
                #     if len(boxes) == 0:
                #         continue
                #     boxes_list.extend(boxes)
                #
                # #
                # # boxes = nms_locality.standard_nms(np.array(boxes_list).astype(np.float64), 0.2)  # Pure python
                # # boxes = nms_locality.nms_locality(np.array(boxes_list).astype(np.float64), 0.2)  # Pure python
                # boxes = lanms.merge_quadrangle_n9(np.array(boxes_list).astype(np.float64), 0.2)             # Python + C++
                # Single output
                boxes, timer = detect(score_map=score[3], geo_map=geometry[3], timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if len(boxes) != 0:
                    boxes_score = boxes[:, 8].reshape((-1, 1))
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                sum_time += duration

                # save to file
                # if len(boxes) != 0:
                if True:
                    res_file = os.path.join(
                        FLAGS.txt_output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        if len(boxes) == 0:
                            continue
                        for idx, box in enumerate(boxes):

                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            box = check_and_validate(box)
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)

                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.img_output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
            print('Average second per frame is {}'.format(sum_time / len(im_fn_list)))
            print('Average frame per second is {}'.format(1.0 / (sum_time / len(im_fn_list))))


if __name__ == '__main__':
    # tf.app.run()
    main()
