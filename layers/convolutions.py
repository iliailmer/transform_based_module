from transform_blocks import HarmonicBlock, HadamardBlock, SlantBlock
from torch.nn import functional as F


class HarmonicConv2d(HarmonicBlock):
    def __init__(self, *args, **kwargs):
        super(HarmonicConv2d, self).__init__(
            output_channels=1, *args, **kwargs
        )
        self.conv = None
        self.bn = None
        self.dropout = None

    def forward(self, x):
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )
        return x


class HarmonicConvTranspose2d(HarmonicBlock):
    def __init__(self, _output_channels=3, *args, **kwargs):
        super(HarmonicConvTranspose2d, self).__init__(
            input_channels=_output_channels, output_channels=1, *args, **kwargs
        )
        self.conv = None
        self.bnorm = None
        self.shortcut = None
        self.dropout = None
        self._output_channels = _output_channels

    def forward(self, x):
        x = F.conv_transpose2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self._output_channels,
        )
        return x


class HadamardConv2d(HadamardBlock):
    def __init__(self, *args, **kwargs):
        super(HadamardConv2d, self).__init__(
            output_channels=1, *args, **kwargs
        )
        self.conv = None
        self.bnorm = None
        self.shortcut = None
        self.dropout = None

    def forward(self, x):
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )
        return x


class HadamardConvTranspose2d(HadamardBlock):
    def __init__(self, _output_channels=3, *args, **kwargs):
        super(HadamardConvTranspose2d, self).__init__(
            input_channels=_output_channels, output_channels=1, *args, **kwargs
        )
        self.conv = None
        self.bnorm = None
        self.shortcut = None
        self.dropout = None
        self._output_channels = _output_channels

    def forward(self, x):
        x = F.conv_transpose2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self._output_channels,
        )
        return x


class SlantConv2d(SlantBlock):
    def __init__(self, *args, **kwargs):
        super(SlantConv2d, self).__init__(output_channels=1, *args, **kwargs)
        self.conv = None
        self.bnorm = None
        self.shortcut = None
        self.dropout = None

    def forward(self, x):
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )
        return x


class SlantConvTranspose2d(SlantBlock):
    def __init__(self, _output_channels=3, *args, **kwargs):
        super(SlantConvTranspose2d, self).__init__(
            input_channels=_output_channels, output_channels=1, *args, **kwargs
        )
        self.conv = None
        self.bnorm = None
        self.shortcut = None
        self.dropout = None
        self._output_channels = _output_channels

    def forward(self, x):
        x = F.conv_transpose2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self._output_channels,
        )
        return x
