import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from ocr.modules.transformation import TPS_SpatialTransformerNetwork
from ocr.modules.feature_extraction import ResNet_FeatureExtractor
from ocr.modules.sequence_modeling import BidirectionalLSTM
from ocr.modules.prediction import Attention


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.stages = {'Trans': 'TPS', 'Feat': 'ResNet',
                       'Seq': 'BiLSTM', 'Pred': 'Attn'}

        """ Transformation TPS"""
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=1)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(1, 512)

        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, 256, 38)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(
            contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction
