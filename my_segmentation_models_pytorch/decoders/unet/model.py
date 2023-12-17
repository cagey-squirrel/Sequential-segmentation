from typing import Optional, Union, List

from my_segmentation_models_pytorch.encoders import get_encoder
from my_segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from .decoder import UnetDecoder

import torch


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        device = None
    ):
        super().__init__()


        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
    

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


    def mutual_attention(self, features):
        '''
        Modifies features in-place
        Applies mutual attention between successive slices
        For example, for slices [0, 1, 2, ...] slice 1 is enriched with information of slices 0 and 2
        '''

        #*features_shallow, features_last_layer = features
        #new_values = torch.zeros(features_last_layer.shape)
        new_features = []
        for features_index, feature_element in enumerate(features[2:]):
            new_values = torch.zeros(feature_element.shape)
            for index in range(1, feature_element.shape[0] - 1):

                features_before  = feature_element[index - 1]
                features_current = feature_element[index]
                features_after   = feature_element[index + 1]

                attention_value = self.attention_modules[features_index](features_before, features_current, features_after)
                new_values[index] = feature_element[index] + attention_value

            new_features.append(new_values)
            #features[-1][index] = features[-1][index] + 5#attention_value # TODO: define attention here
        #return [*features_shallow, features_last_layer]
        return new_features


class MutualAttention(torch.nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        out_channels = max(3, n_channels//8)
        self.query = torch.nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.key = torch.nn.Conv2d(in_channels=n_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.value = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0)
        #self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma_before = torch.nn.Parameter(torch.tensor([0.5]))
        self.gamma_after = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, features_before, features_current, features_after):
        #Notation from the paper.

        channels, width, height = features_current.shape
        query = self.query(features_current)
        keys_before = self.key(features_before)
        keys_after = self.key(features_after)
        values_before = self.value(features_before)
        values_after = self.value(features_after)

        query = query.view(query.shape[0], -1)
        keys_before = keys_before.view(query.shape[0], -1)
        keys_after = keys_after.view(keys_after.shape[0], -1)
        values_before = values_before.view(values_before.shape[0], -1)
        values_after = values_after.view(values_after.shape[0], -1)

        query = query.T

        query = query[None, ...]
        keys_before = keys_before[None, ...]
        keys_after = keys_after[None, ...]
        values_before = values_before[None, ...]
        values_after = values_after[None, ...]
  
        similarity_before = torch.nn.functional.softmax(torch.bmm(query, keys_before), dim=0)
        similarity_after = torch.nn.functional.softmax(torch.bmm(query, keys_after), dim=0)

        attention = self.gamma_before * torch.bmm(values_before, similarity_before) + self.gamma_after * torch.bmm(values_after, similarity_after)
        attention = attention.view(channels, width, height)
 
        return attention