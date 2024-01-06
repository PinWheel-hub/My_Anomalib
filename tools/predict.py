"""Test This script performs inference on the test dataset and saves the output visualizations into a directory."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.deploy import OpenVINOInferencer

from anomalib.data.utils import read_image
from matplotlib import pyplot as plt

import os
import cv2

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="patchcore", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, default="configs/patchcore.yaml", required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")

    return parser


def predict(args: Namespace):
    """Test an anomaly model.

    Args:
        args (Namespace): The arguments from the command line.
    """
    config = get_configurable_parameters(
        model_name=args.model,
        config_path=args.config,
        weight_file=args.weight_file,
    )

    if config.project.seed:
        seed_everything(config.project.seed)

    image_path = config.predict.image_path
    image = read_image(path=image_path)
    output_dir = config.predict.output_dir
    openvino_model_path = config.predict.weight_file
    metadata = config.predict.metadata
    inferencer = OpenVINOInferencer(
        path=openvino_model_path,  # Path to the OpenVINO IR model.
        metadata=metadata,
        device="CPU",  # We would like to run it on an Intel CPU.
    )
    predictions = inferencer.predict(image=image)
    print(predictions.pred_score, predictions.pred_label)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visualize the original image
    predictions.image = cv2.cvtColor(predictions.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, 'image.jpg'), predictions.image)

    # Visualize the raw anomaly maps predicted by the model.
    predictions.anomaly_map = cv2.cvtColor(predictions.anomaly_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, 'anomaly_map.jpg'), predictions.anomaly_map)

    # Visualize the heatmaps, on which raw anomaly map is overlayed on the original image.
    predictions.heat_map = cv2.cvtColor(predictions.heat_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, 'heat_map.jpg'), predictions.heat_map)

    # Visualize the segmentation mask.
    predictions.pred_mask = cv2.cvtColor(predictions.pred_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, 'pred_mask.jpg'), predictions.pred_mask)
    
    # Visualize the segmentation mask with the original image.
    predictions.segmentations = cv2.cvtColor(predictions.segmentations, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, 'segmentations.jpg'), predictions.segmentations)

if __name__ == "__main__":
    args = get_parser().parse_args()
    predict(args)
