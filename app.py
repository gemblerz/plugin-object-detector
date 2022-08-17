import rclpy
from rclpy.node import Node

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse

import cv2
import glob
import time

import logging
# from waggle.plugin import Plugin
from waggle.data.vision import Camera

from demo_interfaces.msg import Heartbeat

class ObjectDetector(Node):
    def __init__(self, args):
        super().__init__('demo_object_detector')
        self.name = self.get_namespace()[1:]
        self.pub_heartbeat = self.create_publisher(Heartbeat, 'state', 10)
        self.pub_objectdetection = self.create_publisher(Heartbeat, 'object', 10)
        self.stream = args.stream

    def run(self):
        self.get_logger().info("Object counter running!")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightfile)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        namesfile = 'detection/coco.names'
        class_names = load_class_names(namesfile)
        self.get_logger().info("Model loaded")
        
        with Camera(self.stream) as camera:
            for sample in camera.stream():
                image = sample.data
                sized = cv2.resize(image, (model.width, model.height))
                # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                boxes = do_detect(model, sized, args.confidence_level, 0.6, use_cuda)
                image, found = plot_boxes_cv2(image, boxes[0], class_names=class_names)
                detection_stats = 'found objects: '
                for object_found, count in found.items():
                    detection_stats += f'{object_found} [{count}] '
                self.get_logger().info(detection_stats)
                # time.sleep(1)
                rclpy.spin_once(self, timeout_sec=0)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='yolov4.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument(
        '-object', dest='object',
        action='append',
        help='Object name to count')
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-confidence-level', dest='confidence_level',
        action='store', default=0.8,
        help='Confidence level [0. - 1.] to filter out result')
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    detector = ObjectDetector(args)
    rclpy.init()
    detector.run()
