from torch import nn
from torch.nn import functional as F
import numpy as np
from layers.blocks import SlantBlock, HarmonicBlock, HadamardBlock
from layers.basic_blocks import BasicBlock, Bottleneck


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, **kwargs):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TransformWRN(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=None,
        out_channels=16,
        depth=10,
        num_classes=10,
        widen_factor=1,
        block=HarmonicBlock,
        bn=True,
        add_noise=False,
        drop=False,
        droprate=0.1,
        alpha_root=None,
        layers=None,
        lmbda=None,
        diag=False,
    ):
        super(TransformWRN, self).__init__()
        nChannels = [
            out_channels,
            out_channels * widen_factor,
            out_channels * 2 * widen_factor,
            out_channels * 4 * widen_factor,
        ]
        assert (depth - 4) % 6 == 0, f"Depth must be 4 (mod 6), got {depth % 6}"
        if layers is None:
            n = [(depth - 4) // 6] * 4
        else:
            n = layers * 4 if len(layers) == 1 else layers
        self.layers = n
        self.lmbda = lmbda
        self.diag = diag
        self.bn = bn
        self.dropout = drop  # extra dropout inside the block
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = in_channels
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert len(block) == 4, f"Length of block list must be 4, {len(block)}"
        self.block = block
        if not isinstance(alpha_root, list):
            self.alpha_root = [alpha_root] * 4
        else:
            if len(alpha_root) == 1:
                self.alpha_root = [alpha_root[0]] * 4
            else:
                assert len(alpha_root) == 4, (
                    f"Length of alpha root list must be 4" + f", got {len(alpha_root)}"
                )
                self.alpha_root = alpha_root

        if not isinstance(add_noise, list):
            self.add_noise = [add_noise] * 4
        else:
            if len(add_noise) == 1:
                self.alpha_root = [add_noise[0]] * 4
            else:
                assert len(add_noise) == 4, (
                    f"Length of alpha root list must be 4," + f" {len(add_noise)}"
                )
                self.add_noise = add_noise

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 4
        else:
            if len(kernel_size) == 1:
                kernel_size = [kernel_size[0]] * 4
            else:
                assert len(kernel_size) == 4, (
                    f"Length of kernel size list "
                    + f"must be 4, got {len(kernel_size)}"
                )
                kernel_size = kernel_size
        self.kernel_size = kernel_size
        pad = [k // 2 for k in kernel_size]
        self.pad = pad

        if block[0] == BasicBlock or block[0] == Bottleneck:
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(
                        self.inplanes,
                        self.inplanes,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    # self.norm_layer(self.inplanes),
                    # nn.ReLU(inplace=True),
                    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                ]
            )
        else:
            self.conv1 = block[0](
                input_channels=in_channels,
                output_ch=nChannels[0],
                bn=self.bn,
                dropout=self.dropout,
                add_noise=self.add_noise[0],
                use_res=False,
                bias=False,
                lmbda=self.lmbda,
                kernel_size=kernel_size[0],
                stride=1,
                pad=pad[0],
                alpha_root=self.alpha_root[0],
            )
        self.drop = nn.Dropout(droprate)  # ORIGINAL=0.1

        self.stack1 = self._make_layer(
            block[1],
            nb_layers=n[1],
            in_planes=nChannels[0],
            out_planes=nChannels[1],
            alpha_root=self.alpha_root[1],
            kernel_size=kernel_size[1],  # //2,
            stride=1,
            pad=pad[1],
        )

        self.stack2 = self._make_layer(
            block[2],
            nb_layers=n[2],
            in_planes=nChannels[1],
            out_planes=nChannels[2],
            alpha_root=self.alpha_root[2],
            kernel_size=kernel_size[2],  # //2,
            stride=2,
            pad=pad[1],
        )

        self.stack3 = self._make_layer(
            block[3],
            nb_layers=n[3],
            in_planes=nChannels[2],
            out_planes=nChannels[3],
            alpha_root=self.alpha_root[3],
            kernel_size=kernel_size[3],
            stride=2,
            pad=pad[3],
        )

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        in_planes,
        out_planes,
        alpha_root,
        nb_layers,
        stride,
        kernel_size,
        pad,
    ):
        if block == BasicBlock or block == Bottleneck:
            return self._make_layer_(block, out_planes, nb_layers, stride)
        strides = [stride] + [1] * (nb_layers - 1)
        use_res = True
        stacking = []
        for st in strides:
            stacking.append(
                block(
                    input_channels=in_planes,
                    output_ch=out_planes,
                    lmbda=self.lmbda,
                    diag=self.diag,
                    add_noise=False,
                    bn=self.bn,
                    use_res=use_res,
                    dropout=self.dropout,
                    # extra dropout inside the block
                    alpha_root=alpha_root,
                    kernel_size=kernel_size,
                    stride=st,
                    pad=pad,
                )
            )
            use_res = True

            if kernel_size == 8:
                kernel_size = kernel_size // 4
                pad = kernel_size // 2
            # stacking.append(nn.Dropout(0.5))
            if in_planes != out_planes:
                in_planes = out_planes
                self.inplanes = out_planes
        return nn.Sequential(*stacking)

    def _make_layer_(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _num_parameters(self, trainable=True):
        k = 0
        all_ = 0
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                if m.weight.requires_grad:
                    all_ += m.weight.size().numel()
            if (
                isinstance(m, HarmonicBlock)
                or isinstance(m, HadamardBlock)
                or isinstance(m, SlantBlock)
            ):
                all_ += m.filter_bank.size().numel()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                if m.weight.requires_grad:
                    k += m.weight.size().numel()
        return k, all_

    def forward(self, x):
        conv1 = self.drop(self.conv1(x))
        stack1 = self.drop(self.stack1(conv1))
        stack2 = self.drop(self.stack2(stack1))
        stack3 = self.drop(self.stack3(stack2))
        bn = self.relu(self.bn1(stack3))
        out = F.avg_pool2d(bn, bn.shape[-1])
        out = self.fc(out.view(-1, self.nChannels))
        return out


class TransformResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=None,
        out_channels=64,
        depth=10,
        num_classes=10,
        block=SlantBlock,
        bn=True,
        add_noise=False,
        drop=False,
        droprate=0.1,
        alpha_root=None,
        layers=None,
        lmbda=None,
        diag=False,
    ):
        super(TransformResNet, self).__init__()
        nChannels = [out_channels, out_channels, out_channels * 2, out_channels * 4]
        assert (depth - 4) % 6 == 0, f"Depth must be 4 (mod 6), got {depth % 6}"
        if layers is None:
            n = [(depth - 4) // 6] * 4
        else:
            n = layers * 4 if len(layers) == 1 else layers
        self.layers = n
        self.lmbda = lmbda
        self.diag = diag
        self.bn = bn
        self.dropout = drop  # extra dropout inside the block
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = in_channels
        self.dilation = 1  # parameter for basic/bottleneck
        self.groups = 1  # parameter for basic/bottleneck
        self.base_width = 64
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert len(block) == 4, f"Length of block list must be 4, {len(block)}"
        self.block = block
        if not isinstance(alpha_root, list):
            self.alpha_root = [alpha_root] * 4
        else:
            if len(alpha_root) == 1:
                self.alpha_root = [alpha_root[0]] * 4
            else:
                assert len(alpha_root) == 4, (
                    f"Length of alpha root list must be 4" + f", got {len(alpha_root)}"
                )
                self.alpha_root = alpha_root

        if not isinstance(add_noise, list):
            self.add_noise = [add_noise] * 4
        else:
            if len(add_noise) == 1:
                self.alpha_root = [add_noise[0]] * 4
            else:
                assert len(add_noise) == 4, (
                    f"Length of alpha root list must be 4" + f", {len(add_noise)}"
                )
                self.add_noise = add_noise

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 4
        else:
            if len(kernel_size) == 1:
                kernel_size = [kernel_size[0]] * 4
            else:
                assert len(kernel_size) == 4, (
                    f"Length of kernel size list must be" + f" 4, {len(kernel_size)}"
                )
                kernel_size = kernel_size
        self.kernel_size = kernel_size
        pad = [k // 2 for k in kernel_size]

        if block[0] == BasicBlock or block[0] == Bottleneck:
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(
                        self.inplanes,
                        self.inplanes,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=False,
                    ),
                    self.norm_layer(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ]
            )
        else:
            self.conv1 = block[0](
                input_channels=in_channels,
                bn=self.bn,
                dropout=self.dropout,
                output_ch=in_channels,
                add_noise=self.add_noise[0],
                use_res=False,
                bias=False,
                lmbda=self.lmbda,
                kernel_size=kernel_size[0],
                stride=1,
                pad=pad[0],
                alpha_root=self.alpha_root[0],
            )
        self.drop = nn.Dropout(droprate)  # ORIGINAL=0.1

        self.stack0 = self._make_layer(
            block[0],
            nb_layers=n[0],
            in_planes=in_channels,
            out_planes=nChannels[0],
            alpha_root=self.alpha_root[0],
            kernel_size=kernel_size[0],  # //2,
            stride=1,
            pad=pad[0],
        )

        self.stack1 = self._make_layer(
            block[1],
            nb_layers=n[1],
            in_planes=nChannels[0],
            out_planes=nChannels[1],
            alpha_root=self.alpha_root[1],
            kernel_size=kernel_size[1],  # //2,
            stride=2,
            pad=pad[1],
        )

        self.stack2 = self._make_layer(
            block[2],
            nb_layers=n[2],
            in_planes=nChannels[1],
            out_planes=nChannels[2],
            alpha_root=self.alpha_root[2],
            kernel_size=kernel_size[2],
            stride=2,
            pad=pad[2],
        )

        self.stack3 = self._make_layer(
            block[3],
            nb_layers=n[3],
            in_planes=nChannels[2],
            out_planes=nChannels[3],
            alpha_root=self.alpha_root[3],
            kernel_size=kernel_size[3],
            stride=2,
            pad=pad[3],
        )

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        in_planes,
        out_planes,
        alpha_root,
        nb_layers,
        stride,
        kernel_size,
        pad,
    ):
        if block == BasicBlock or block == Bottleneck:
            return self._make_layer_(block, out_planes, nb_layers, stride)
        strides = [stride] + [1] * (nb_layers - 1)
        use_res = True
        stacking = []
        for st in strides:
            stacking.append(
                block(
                    input_channels=in_planes,
                    output_ch=out_planes,
                    lmbda=self.lmbda,
                    diag=self.diag,
                    add_noise=False,
                    bn=self.bn,
                    use_res=use_res,
                    dropout=self.dropout,
                    # extra dropout inside the block
                    alpha_root=alpha_root,
                    kernel_size=kernel_size,
                    stride=st,
                    pad=pad,
                )
            )
            use_res = True  # resnet always has identity mapping
            if kernel_size == 8:
                kernel_size = kernel_size // 4
                pad = kernel_size // 2
            # stacking.append(nn.Dropout(0.5))
            if in_planes != out_planes:
                in_planes = out_planes
                self.inplanes = out_planes
        return nn.Sequential(*stacking)

    def _make_layer_(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _num_parameters(self, trainable=True):
        k = 0
        all_ = 0
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                if m.weight.requires_grad:
                    all_ += m.weight.size().numel()
            if (
                isinstance(m, HarmonicBlock)
                or isinstance(m, HadamardBlock)
                or isinstance(m, SlantBlock)
            ):
                all_ += m.filter_bank.size().numel()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                if m.weight.requires_grad:
                    k += m.weight.size().numel()
        return k, all_

    def forward(self, x):
        conv1 = self.drop(self.conv1(x))

        stack0 = self.drop(self.stack0(conv1))
        stack1 = self.drop(self.stack1(stack0))
        stack2 = self.drop(self.stack2(stack1))
        stack3 = self.drop(self.stack3(stack2))

        bn = self.relu(self.bn1(stack3))

        out = F.avg_pool2d(bn, bn.shape[-1])

        out = self.fc(out.view(-1, self.nChannels))
        return out


class ResNeXt(nn.Module):
    def __init__(
        self,
        Block,
        num_blocks,
        transform_conv,
        cardinality,
        bottleneck_width,
        num_classes=10,
    ):
        super(ResNeXt, self).__init__()
        self.block = Block
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], transform_conv, 1)
        self.layer2 = self._make_layer(num_blocks[1], transform_conv, 2)
        self.layer3 = self._make_layer(num_blocks[2], transform_conv, 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, transform_conv, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                self.block(
                    self.in_planes,
                    self.cardinality,
                    transform_conv,
                    self.bottleneck_width,
                    stride,
                )
            )
            self.in_planes = (
                self.block.expansion * self.cardinality * self.bottleneck_width
            )
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d(conv, block):
    return ResNeXt(
        num_blocks=[3, 3, 3],
        Block=block,
        cardinality=2,
        transform_conv=conv,
        bottleneck_width=64,
    )


def ResNeXt29_4x64d(conv, block):
    return ResNeXt(
        num_blocks=[3, 3, 3],
        Block=block,
        transform_conv=conv,
        cardinality=4,
        bottleneck_width=64,
    )


def ResNeXt29_8x64d(conv, block):
    return ResNeXt(
        num_blocks=[3, 3, 3],
        Block=block,
        transform_conv=conv,
        cardinality=8,
        bottleneck_width=64,
    )


def ResNeXt29_32x4d(conv, block):
    return ResNeXt(
        num_blocks=[3, 3, 3],
        Block=block,
        transform_conv=conv,
        cardinality=32,
        bottleneck_width=4,
    )
