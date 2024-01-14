import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        super(Conv2dReLU, self).__init__()

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        self.relu = nn.ReLU(inplace=True)
        self.use_batchnorm = use_batchnorm

        if use_batchnorm == "inplace":
            self.bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            self.relu = nn.Identity()
        
        elif use_batchnorm and use_batchnorm == "layer_norm":
            self.bn = LayerNormalization(out_channels)

        elif use_batchnorm and use_batchnorm != "inplace":
            self.bn = nn.BatchNorm2d(out_channels)

        else:
            self.bn = nn.Identity()
    
        #super(Conv2dReLU, self).__init__(conv, bn, relu)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LayerNormalization(nn.Module):
    '''
    Custom implementation of Layer Normalization Module
    I need to normalize data on channel dimension
    To do this, I first need to permute the dimensions so the C dimension is last
    B, C, W, H --> B, W, H, C
    Then normalize on channel dimension
    And permute the dimensions back: B, W, H, C --> B, C, W, H
    '''
    def __init__(
        self,
        out_channels
    ):
        super(LayerNormalization, self).__init__()
        print("Layer Normalization")
        image_size = 256//out_channels * 14
        
        
        self.ln = nn.LayerNorm((image_size, image_size))
        #self.ln = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        #x = torch.permute(x, (0, 2, 3, 1))
        x = self.ln(x)
        #x = torch.permute(x, (0, 3, 1, 2))
        #print(f'passed forward')
        return x

#24436369
#24823249

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class AttentionModule(nn.Module):
        '''
        Modifies features in-place
        Applies mutual attention between successive slices
        For example, for slices [0, 1, 2, ...] slice 1 is enriched with information of slices 0 and 2
        '''

        def __init__(self, in_channels) -> None:
            super().__init__()
            print(f'Sprinkle in decoder attention for {in_channels} channels')
            #self.self_attention = SelfAttention(channels)MutualAttention
            self.self_attention = MutualAttention(in_channels)

        def forward(self, features):
            return self.forward2(features)
        

        def forward1(self, features):


            new_features = []
     
            for index in range(features.shape[0]):
                features_current = features[index]

                if index == 0:
                    features_before = features_current
                else:
                    features_before  = features[index - 1]

                if index == features.shape[0] - 1:
                    features_after = features_current
                else:
                    features_after   = features[index + 1]



                new_feature = self.self_attention(features_before, features_current, features_after)
                new_features.append(new_feature)
            
            return torch.stack(new_features)

        def forward2(self, features):


            new_features = []
     
            for index in range(features.shape[0]):
                features_current = features[index]

                if index == 0:
                    features_before = features_current
                else:
                    features_before  = features[index - 1]

                if index == features.shape[0] - 1:
                    features_after = features_current
                else:
                    features_after   = features[index + 1]



                new_feature = self.self_attention(features_before, features_current, features_after)
                new_features.append(new_feature)
            
            return torch.stack(new_features)


class MutualAttention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, 
                 dimension=2, bn_layer=False):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(MutualAttention, self).__init__()


        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
       
        conv_nd = nn.Conv2d
        #max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d
        

        

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # function g in the paper which goes through conv. with kernel size 1
        self.queries = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.keys    = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.values  = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        self.gamma_before = nn.Parameter(torch.tensor([0.]))
        self.gamma_after  = nn.Parameter(torch.tensor([0.]))
       
            
    def forward(self, features_before, features_current, features_after):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        channels, width, height = features_current.shape
        
        
        query = self.queries(features_current).view(self.inter_channels, -1)
        query = query.permute(1, 0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        values_before = self.values(features_before).view(self.inter_channels, -1)
        values_before = values_before.permute(1, 0)
        key_before = self.keys(features_before).view(self.inter_channels, -1)
        
        f_before = torch.matmul(query, key_before)

        f_div_C_before = torch.nn.functional.softmax(f_before, dim=-1)

        
        y_before = torch.matmul(f_div_C_before, values_before)
        
        # contiguous here just allocates contiguous chunk of memory
        y_before = y_before.permute(1, 0).contiguous()
        y_before = y_before.view(self.inter_channels, width, height)
        
        W_y_before = self.W_z(y_before)


        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        values_after = self.values(features_after).view(self.inter_channels, -1)
        values_after = values_after.permute(1, 0)
        key_after = self.keys(features_after).view(self.inter_channels, -1)
        
        f_after = torch.matmul(query, key_after)

        f_div_C_after = torch.nn.functional.softmax(f_after, dim=-1)

        
        y_after = torch.matmul(f_div_C_after, values_after)
        
        # contiguous here just allocates contiguous chunk of memory
        y_after = y_after.permute(1, 0).contiguous()
        y_after = y_after.view(self.inter_channels, width, height)
        
        W_y_after = self.W_z(y_after)


        # residual connection
        z = W_y_before + W_y_after + features_current
        #print(f'z.shape = {z.shape}')

        return z


class Attention(nn.Module):
    def __init__(self, name, attention_oder='', **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        elif name == "single-scse":  # For single scse we only apply attention to second layer
            if attention_oder == 'first':
                self.attention = nn.Identity(**params)
            elif attention_oder == 'second':
                self.attention = SCSEModule(**params)
        elif name == "ma":
            if attention_oder == 'first':
                self.attention = nn.Identity(**params)
            elif attention_oder == 'second':
                self.attention = AttentionModule(**params)
            
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)
