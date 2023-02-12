import mnn
import torch_geometric


class FeaStResBlock(mnn.ResBlock):

    def __init__(self, in_channels, out_channels, **kwargs):
        convolution = torch_geometric.nn.FeaStConv

        super(FeaStResBlock, self).__init__(convolution, in_channels, out_channels, **kwargs)
