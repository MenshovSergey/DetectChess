import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from os2d.modeling.model import build_os2d_from_config
from os2d.config import cfg
import os2d.utils.visualization as visualizer
from os2d.structures.feature_map import FeatureMapSize
from os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio
import os
logger = setup_logger("OS2D")


class OneShotDetector():
    def __init__(self, class_images, class_ids, path_to_model, is_cuda=True):
        # use GPU if have available
        # cfg.is_cuda = torch.cuda.is_available()
        cfg.is_cuda = is_cuda

        cfg.init.model = path_to_model
        self.net, self.box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(img_normalization["mean"], img_normalization["std"])
        ])

        self.class_ids = class_ids
        self.class_images_th = []
        for class_image in class_images:
            h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
                                                                       w=class_image.size[0],
                                                                       target_size=cfg.model.class_image_size)
            class_image = class_image.resize((w, h))

            class_image_th = self.transform_image(class_image)
            if cfg.is_cuda:
                class_image_th = class_image_th.cuda()

            self.class_images_th.append(class_image_th)

    def detect(self, input_image, lower=True, max_parabola_deviation=0.05):
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                                   w=input_image.size[0],
                                                                   target_size=256)
        input_image = input_image.resize((w, h))

        input_image_th = self.transform_image(input_image)
        input_image_th = input_image_th.unsqueeze(0)
        if cfg.is_cuda:
            input_image_th = input_image_th.cuda()

        # with torch.no_grad():
        #     feature_map = net.net_feature_maps(input_image_th)

        #     class_feature_maps = net.net_label_features(class_images_th)
        #     class_head = net.os2d_head_creator.create_os2d_head(class_feature_maps)

        #     loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(class_head=class_head,
        #                                                                                             feature_maps=feature_map)

        with torch.no_grad():
            loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = self.net(
                images=input_image_th, class_images=self.class_images_th)

        image_loc_scores_pyramid = [loc_prediction_batch[0]]
        image_class_scores_pyramid = [class_prediction_batch[0]]
        transform_corners_pyramid = [transform_corners_batch[0]]

        img_size_pyramid = [FeatureMapSize(img=input_image_th)]

        cfg.visualization.eval.max_detections = 100
        cfg.visualization.eval.score_threshold = 0.6
        cfg.visualization.eval.area_threshold = 1

        boxes = self.box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                              img_size_pyramid, self.class_ids,
                                              nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                              nms_score_threshold=cfg.eval.nms_score_threshold,
                                              area_threshold=cfg.visualization.eval.area_threshold,
                                              image_area=input_image.height * input_image.width,
                                              transform_corners_pyramid=transform_corners_pyramid)

        # remove some fields to lighten visualization
        boxes.remove_field("default_boxes")

        # Note that the system outputs the correaltions that lie in the [-1, 1] segment as the detection scores (the higher the better the detection).
        scores = boxes.get_field("scores")

        figsize = (8, 8)
        fig = plt.figure(figsize=figsize)
        columns = len(class_images)
        for i, class_image in enumerate(class_images):
            fig.add_subplot(1, columns, i + 1)
            plt.imshow(class_image)
            plt.axis('off')

        plt.rcParams["figure.figsize"] = figsize

        return visualizer.show_detections(boxes, input_image,
                                   cfg.visualization.eval,
                                   lower=lower, max_parabola_deviation=max_parabola_deviation)


if __name__ == '__main__':
    path_dataset = '/home/sergej/idchess/images'
    files = os.listdir(path_dataset)
    class_images = []
    class_ids = []
    for v in files:
        class_images.append(read_image(os.path.join(path_dataset, v)))
        class_ids.append(0)


    path_to_model = "models/os2d_v2-train.pth"
    detector = OneShotDetector(class_images, class_ids, path_to_model)
    files = os.listdir("/home/sergej/idchess/test")
    for v in files:
        print(v)
        input_image = read_image(os.path.join("/home/sergej/idchess/test", v))
        res = detector.detect(input_image, lower=True, max_parabola_deviation=None)
        res.savefig(os.path.join("/home/sergej/idchess/visualize", v))
