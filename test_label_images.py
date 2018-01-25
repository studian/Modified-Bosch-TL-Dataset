#!/usr/bin/env python
"""
Quick sample script that displays the traffic light labels within
the given images.
If given an output folder, it draws them to file.

Example usage:
    python write_label_images input.yaml [output_folder]
"""
import sys
import os
import cv2
from read_label_file import get_all_labels
from data_utils import load_tl_extracts, load_tl_extracts_hkkim
import numpy as np
from tl_classifier_cnn import TLClassifierCNN, TLLabelConverter
import argparse


def ir(some_value):
    """Int-round function for short array indexing """
    return int(round(some_value))


def show_label_images(input_yaml, wait_ms=10):
    """
    Shows and draws pictures with labeled traffic lights.
    Can save pictures.

    :param input_yaml: Path to yaml file
    :param wait_ms: wait time in milliseconds before OpenCV shows next image
    """
    label_list = {'off':0, 'green':1, 'yellow':2, 'red':3}

    # load the model
    tlc = TLClassifierCNN()
    model_dir = 'model'
    tlc.load_model(model_dir)

    # Shows and draws pictures with labeled traffic lights
    images = get_all_labels(input_yaml)

    for i, image_dict in enumerate(images):
        image = cv2.imread(image_dict['path'])
        if image is None:
            raise IOError('Could not open image path', image_dict['path'])
            break

        for box in image_dict['boxes']:
            xmin = ir(box['x_min'])
            ymin = ir(box['y_min'])
            xmax = ir(box['x_max'])
            ymax = ir(box['y_max'])

            if xmax-xmin<=0 or ymax-ymin<=0:
                continue

            label = box['label']
            label = label.lower()

            roi = image[ymin:(ymax+1), xmin:(xmax+1)]
            resized_roi = cv2.resize(roi, (32,32), interpolation=cv2.INTER_LINEAR)
            prd_labels, prd_probabilities = tlc.predict(np.array([resized_roi]), batch_size=1)
            prd_prob = prd_probabilities[0][label_list[prd_labels[0]]] * 100

            if label == prd_labels[0]:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0))
                label_str = '%s(%.2f)' % (prd_labels[0], prd_prob)
                image = cv2.putText(image, label_str, (xmin, ymax+20), 0, 0.4, (0,255,0), 1, cv2.LINE_AA) # text green
 
            else:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255)) # color red
                label_str = '%s: %s(%.2f)' % (label, prd_labels[0], prd_prob)
                image = cv2.putText(image, label_str, (xmin, ymax+20), 0, 0.4, (0,0,255), 1, cv2.LINE_AA)
           

        cv2.imshow('labeled_image', image)
        #cv2.waitKey(10)

        if cv2.waitKey(wait_ms) == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_yaml")

    args = parser.parse_args()
    input_yaml = args.input_yaml

    wait_ms = 50
    show_label_images(input_yaml, wait_ms=wait_ms)
