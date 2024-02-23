from scripts.nudenet_nsfw_censor_scripts.pil_nude_detector import pil_nude_detector, nudenet_labels_index, mask_shapes_func_dict
from scripts.nudenet_nsfw_censor_scripts.censor_image_filters import apply_filter, filter_dict
from modules.api.api import decode_base64_to_image, encode_pil_to_base64
from fastapi import FastAPI, Body
from PIL import ImageFilter, Image
from modules import shared
from math import sqrt
import gradio as gr
import numpy as np
import base64


def nudenet_censor_api(_: gr.Blocks, app: FastAPI):
    @app.post("/nudenet/detect")
    async def detect_nsfw(
            input_image: str = Body(None, title="base64 input image"),
            mask_shape: str = Body(None, title=f"Name Mask shape: {list(mask_shapes_func_dict)}"),
            nms_threshold: float = Body(None, title="NMS threshold: float [0, 1]"),
            rectangle_round_radius: float = Body(None, title="Rectangle round radius: float [0, inf]"),
            thresholds: list = Body(None, title=f"list of float for thresholds of: {list(nudenet_labels_index)}"),
            expand_horizontal: list = Body(None, title=f"List of float for expand horizontal: {list(nudenet_labels_index)}"),
            expand_vertical: list = Body(None, title=f"List of float for expand vertical: {list(nudenet_labels_index)}"),
    ):
        input_image = decode_base64_to_image(input_image)
        if not input_image:
            return {'nsfw_detected': False, 'error': 'Invalid image'}

        nms_threshold = nms_threshold if nms_threshold else shared.opts.nudenet_nsfw_censor_nms_threshold
        mask_shape = mask_shape if mask_shape else shared.opts.nudenet_nsfw_censor_mask_shape
        rectangle_round_radius = rectangle_round_radius if rectangle_round_radius else shared.opts.nudenet_nsfw_censor_rectangle_round_radius
        if pil_nude_detector.thresholds is None:
            pil_nude_detector.refresh_label_configs()

        thresholds = np.asarray(thresholds) if thresholds else pil_nude_detector.thresholds
        expand_horizontal = np.asarray(expand_horizontal) if expand_horizontal else pil_nude_detector.expand_horizontal
        expand_vertical = np.asarray(expand_vertical) if expand_vertical else pil_nude_detector.expand_vertical

        try:
            nudenet_mask = pil_nude_detector.get_censor_mask(input_image, nms_threshold, mask_shape, rectangle_round_radius, thresholds, expand_horizontal, expand_vertical).convert('L')
            nsfw_detected = bool(np.any(np.array(nudenet_mask)))
            return {'nsfw_detected': nsfw_detected}
        except Exception as e:
            return {'nsfw_detected': False, 'error': str(e)}

