import torch
import mnn
from utils import parameter_table
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F


class MeshGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, improved=False,
                 cached=False, bias=True, fine_marker_dict=None):
        super().__init__()

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            convs.append(GCNConv(channels[i], channels[i+1], improved=improved,
                                 cached=cached, bias=bias))
        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        tmp = [data.featr3[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        data.featr3[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        data.featr3[:, 2].reshape(-1, 9)]
        data.x = torch.hstack((torch.hstack(tmp), data.stmdist))
        
        
        for i, conv in enumerate(self.convs[:-1]):
            out = conv(data.x, data.edge_index)
            out = F.relu(out)

        out = self.convs[-1](out, data.edge_index)
        return out










# Base class for different convolutions
class Compare(torch.nn.Module):
    def __init__(self):
        super(Compare, self).__init__()

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table.create(self)

    def forward(self, data):
        # (N, 3, 3, 3) -> [(N, 6), (N, 6), (N, 9)] (two matrices are symmetric)
        tmp = [data.featr3[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
               data.featr3[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
               data.featr3[:, 2].reshape(-1, 9)]
        data.x = torch.hstack((torch.hstack(tmp), data.stmdist))

        # Encoder
        data.x = self.conv01(data.x, data.scale0_edge_index)
        data.x = self.conv02(data.x, data.scale0_edge_index)

        # Downstream
        copy0 = data.x.clone()
        data = self.pool1(data)
        data.x = self.conv11(data.x, data.edge_index)
        data.x = self.conv12(data.x, data.edge_index)

        copy1 = data.x.clone()
        data = self.pool2(data)
        data.x = self.conv21(data.x, data.edge_index)
        data.x = self.conv22(data.x, data.edge_index)

        # Upstream
        data = self.pool2.unpool(data)
        data.x = torch.cat((data.x, copy1), dim=1)  # "copy/cat"
        data.x = self.conv13(data.x, data.edge_index)
        data.x = self.conv14(data.x, data.edge_index)
        data.x = self.conv15(data.x, data.edge_index)
        data.x = self.conv16(data.x, data.edge_index)

        # Decoder
        data = self.pool1.unpool(data)
        data.x = torch.cat((data.x, copy0), dim=1)  # "copy/cat"
        data.x = self.conv03(data.x, data.edge_index)
        data.x = self.conv04(data.x, data.edge_index)
        data.x = self.conv05(data.x, data.edge_index)
        data.x = self.conv06(data.x, data.edge_index)

        return data.x


# FeaSt convolutional residual network for comparison with the GEM-CNN
class CompareFeaSt(Compare):
    def __init__(self,out_channel = 3):
        super(CompareFeaSt,  self).__init__()

        channels = 115
        heads = 2

        # Encoder
        self.conv01 = mnn.FeaStResBlock(22, channels, heads=heads)
        self.conv02 = mnn.FeaStResBlock(channels, channels, heads=heads)

        # Downstream
        self.pool1 = mnn.ClusterPooling(1)
        self.conv11 = mnn.FeaStResBlock(channels, channels, heads=heads)
        self.conv12 = mnn.FeaStResBlock(channels, channels, heads=heads)

        self.pool2 = mnn.ClusterPooling(2)
        self.conv21 = mnn.FeaStResBlock(channels, channels, heads=heads)
        self.conv22 = mnn.FeaStResBlock(channels, channels, heads=heads)

        # Up-stream
        self.conv13 = mnn.FeaStResBlock(channels + channels, channels, heads=heads)
        self.conv14 = mnn.FeaStResBlock(channels, channels, heads=heads)
        self.conv15 = mnn.FeaStResBlock(channels, channels, heads=heads)
        self.conv16 = mnn.FeaStResBlock(channels, channels, heads=heads)

        # Decoder
        self.conv03 = mnn.FeaStResBlock(channels + channels, channels, heads=heads)
        self.conv04 = mnn.FeaStResBlock(channels, channels, heads=heads)
        self.conv05 = mnn.FeaStResBlock(channels, channels, heads=heads)
        self.conv06 = mnn.FeaStResBlock(channels, out_channel, heads=heads, relu=False)

        print(self.parameter_table())


# FeaSt convolutional residual network for comparison with the GEM-CNN
class CompareSAGE(Compare):
    def __init__(self,out_channel = 3):
        super(CompareSAGE, self).__init__()

        # channels = 116
        channels = 128

        # Encoder
        self.conv01 = mnn.SAGEResBlock(24, channels)
        self.conv02 = mnn.SAGEResBlock(channels, channels)

        # Downstream
        self.pool1 = mnn.ClusterPooling(1)
        self.conv11 = mnn.SAGEResBlock(channels, channels)
        self.conv12 = mnn.SAGEResBlock(channels, channels)

        self.pool2 = mnn.ClusterPooling(2)
        self.conv21 = mnn.SAGEResBlock(channels, channels)
        self.conv22 = mnn.SAGEResBlock(channels, channels)

        # Up-stream
        self.conv13 = mnn.SAGEResBlock(channels + channels, channels)
        self.conv14 = mnn.SAGEResBlock(channels, channels)
        self.conv15 = mnn.SAGEResBlock(channels, channels)
        self.conv16 = mnn.SAGEResBlock(channels, channels)

        # Decoder
        self.conv03 = mnn.SAGEResBlock(channels + channels, channels)
        self.conv04 = mnn.SAGEResBlock(channels, channels)
        self.conv05 = mnn.SAGEResBlock(channels, channels)
        self.conv06 = mnn.SAGEResBlock(channels, out_channel, relu=False)

        print(self.parameter_table())
