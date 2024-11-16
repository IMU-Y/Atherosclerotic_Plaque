import torch
import torch.nn as nn
import math


# ----------------------------#
# SE注意力机制
# ----------------------------#
class SE_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_block, self).__init__()
        # --------------------------------------------------#
        # 此为自适应的二维平均全局池化操作
        # 通道数不会发生改变
        # The output is of size H x W, for any input size.
        # AdaptiveAvgPool2d(1) = AdaptiveAvgPool2d((1,1))
        # --------------------------------------------------#
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # -------------------------------------------------------#
        # 对于全连接层,第一个参数为输入的通道数,第二个参数为输入的通道数
        # 之后经过ReLU来提升模型的非线性表达能力,以及对特征信息进行编码
        # sigmoid还是一个激活函数,来提升模型的非线性表达能力
        # ratio越大其对于特征融合以及信息表达所产生的影响越大,
        # 压缩降维对于学习通道之间的依赖关系有着不利影响
        # -------------------------------------------------------#
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # -----------------------------------------------------------------#
        # 先进行自适应二维全局平均池化,然后进行一个reshape的操作
        # 之后使其通过一个全连接层、一个ReLU、一个全连接层、一个Sigmoid层
        # 再将其reshape成之前的shape即可
        # 最后将注意力权重y和输入X按照通道加权相乘，调整模型对输入x不同通道的重视程度
        # ------------------------------------------------------------=----#
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
