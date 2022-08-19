import rclpy
from rclpy.node import Node

from tool.utils import *
from tool.torch_utils import *
from hubconf import nvidia_ssd_processing_utils
import torch
import argparse

import cv2
import glob
import time

import logging
# from waggle.plugin import Plugin
from waggle.data.vision import Camera

from std_msgs.msg import String
from demo_interfaces.msg import Heartbeat

class ObjectDetector(Node):
    def __init__(self, args):
        super().__init__('demo_object_detector')
        self.name = self.get_namespace()[1:]
        self.pub_heartbeat = self.create_publisher(Heartbeat, 'state', 10)
        self.pub_objectdetection = self.create_publisher(String, '/object', 10)
        self.stream = args.stream
        self.conf_level = args.confidence_level

    def run(self):
        self.get_logger().info("Object counter running!")
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            ssd_model = torch.load(args.model)
            ssd_model.to('cuda')
        else:
            ssd_model = torch.load(args.model, map_location=torch.device('cpu'))
        ssd_model.eval()
        utils = nvidia_ssd_processing_utils()
        classes_to_labels = utils.get_coco_object_dictionary()
        self.get_logger().info("Model loaded")
        
        with Camera(self.stream) as camera:
            for sample in camera.stream():
                image = sample.data
                height = image.shape[0]
                width = image.shape[1]
                timestamp = sample.timestamp
                inputs = [utils.prepare_input(None, image, image_size=(300, 300))]
                tensor = utils.prepare_tensor(inputs, cuda=use_cuda)
                with torch.no_grad():
                    detections_batch = ssd_model(tensor)
                results_per_input = utils.decode_results(detections_batch)
                best_results_per_input = [utils.pick_best(results, self.conf_level) for results in results_per_input]
                bboxes, classes, confidences = best_results_per_input[0]
                for box, cls, conf in zip(bboxes, classes, confidences):
                    object_label = classes_to_labels[cls]
                    detection_stats = f'{timestamp/1e9}: found objects: {object_label} ({conf})'
                    self.get_logger().info(detection_stats)
                    msg = String()
                    msg.data = object_label
                    self.pub_objectdetection.publish(msg)
                # time.sleep(1)
                rclpy.spin_once(self, timeout_sec=0)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-model', type=str, default='coco_ssd_resnet50_300_fp32.pth',
                        help='path of modelfile')
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
    rclpy.init()
    detector = ObjectDetector(args)
    detector.run()
