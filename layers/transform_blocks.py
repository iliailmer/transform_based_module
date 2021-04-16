import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard, block_diag

__all__ = ["HarmonicBlock", "HadamardBlock", "SlantBlock"]


class TransformBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        bn=True,
        dropout=False,
        kernel_size=3,
        padding=1,
        stride=1,
        alpha_root=None,
        add_noise=False,
        lmbda=None,
        diag=False,
        bias=False,
        use_res=True,
    ):
        super().__init__()
        self._cuda = torch.cuda.is_available()
        self.bn = bn  # flag for batchnorm (True/False)
        self.drop = dropout  # flag for dropout (True/False)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.bias = bias  # flag for bias
        self.diag = diag  # select only diagonal filters
        self.use_res = use_res  # use residual connection or not
        self.alpha_root = alpha_root  # use rooting, i.e. (input.pow(alpha))
        # Filters based on transform
        self.filter_bank = None
        # Linear Combination conv
        self.conv = None
        # Shortcut is the downsampling layer. It preserves the input info
        self.shortcut = nn.Identity()
        # if use_res is False,
        # it is just identity mapping
        if (input_channels != output_channels or stride != 1) and self.use_res:
            ks = 2 if self.kernel_size % 2 == 0 else 1
            p = 1 if ks == 2 else 0
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    padding=p,
                    kernel_size=ks,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )

        if add_noise:
            self.noise = nn.Parameter(
                nn.init.normal_(
                    torch.randn(1, self.filter_bank.size(0)), std=1e-4
                )
            )
        else:
            self.noise = None
        if lmbda is not None:
            # limits the number of kernels
            self.lmbda = min(lmbda, self.kernel_size ** 2)
        else:
            self.lmbda = lmbda

    """
    Base class for any tranform-based layer.
    Main idea: compute window-based transform from each channel,
    combine linearly through 1x1 convolution.
    Class allows use of residual connection as in ResNets.
    """

    @classmethod
    def get_idx(self, ker: int, lmbda: int):
        """Return indices for partial filter usage, see Ulicny et al. 2018."""
        out = []
        for i in range(ker):
            for j in range(ker):
                if i + j < lmbda:
                    out.append(ker * i + j)
        return tuple(out)

    @classmethod
    def get_idx_diag(self, ker):
        """Return only diagonal indices for partial filter usage."""
        out = []
        for i in range(ker):
            for j in range(ker):
                if i == j:
                    out.append(i + j)
        return tuple(out)

    @classmethod
    def draw_filters(self, fb_=None, figsize=(12, 4)):
        """Display visual representation of all filters as a grid.

        Parameters:
            fb_: any filter bank to display. If `None` is passed,
                 the default filter bank of the class is shown.
                 This argument must comply with PyTorch's weight shapes.
            figsize: a 2-tuple of integers, compatible with `figsize` of
                 `matplotlib.pyplot`
        """
        if not fb_:
            fb_ = self.filter_bank
        fig, ax = plt.subplots(len(fb_), 1, figsize=figsize)
        for i in range(len(fb_)):
            ax[i].imshow(fb_[i, 0, :, :])
            ax[i].axis("off")
            ax[i].grid(False)

    def filter_from_matrix(self, i, j, size):
        raise NotImplementedError

    def get_filter_bank(
        self, kernel_size, input_channels=3, lmbda=None, diag=False, **kwargs
    ):
        """Build filter bank from a matrix.

        Parameters:
             :kernel_size: integer, determines sizes of filters.
             input
        """
        filter_bank = torch.zeros(
            (kernel_size, kernel_size, kernel_size, kernel_size)
        )
        for i in range(kernel_size):
            for j in range(kernel_size):
                filter_bank[i, j, :, :] = self.filter_from_matrix(
                    i=i, j=j, size=kernel_size
                )
        if lmbda:
            ids = self.get_idx(kernel_size, lmbda)
            return torch.stack(
                tuple(
                    [
                        (filter_bank.view(-1, 1, kernel_size, kernel_size))[
                            ids, ...
                        ]
                    ]
                    * input_channels
                ),
                dim=0,
            ).view((-1, 1, kernel_size, kernel_size))
        if diag:
            ids = self.get_idx_diag(kernel_size)
            return torch.stack(
                tuple(
                    [
                        filter_bank.view(-1, 1, kernel_size, kernel_size)[
                            ids, ...
                        ]
                    ]
                    * input_channels
                ),
                dim=0,
            ).view((-1, 1, kernel_size, kernel_size))
        return torch.stack(
            tuple(
                [filter_bank.view(-1, 1, kernel_size, kernel_size)]
                * input_channels
            ),
            dim=0,
        ).view((-1, 1, kernel_size, kernel_size))

    @classmethod
    def alpha_rooting(self, x, alpha=1.0):
        if alpha is not None:
            return x.sign() * torch.abs(x).pow(alpha)
        else:
            return x


class HarmonicBlock(TransformBlock):
    def __init__(
        self,
        *args,
        type=2,
        droprate=0.5,
        **kwargs,
    ):
        super(HarmonicBlock, self).__init__(*args, **kwargs)
        self.type = type
        self.filter_bank = self.get_filter_bank(
            kernel_size=self.kernel_size,  # kernel size
            input_channels=self.input_channels,
            t=self.type,  # type of DCT
            lmbda=self.lmbda,
            diag=self.diag,
        ).float()
        if self._cuda:
            self.filter_bank = self.filter_bank.cuda()
        self.conv = nn.Conv2d(
            in_channels=self.filter_bank.shape[0],
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=self.bias,
        )
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(droprate)

    @staticmethod
    def dct_matrix(t=2, N=32):
        if t == 1:
            # N- the size of the input
            # n is the column dummy index, k is the row dummy index
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for p in range(N):
                res[p, -1] = (-1) ** p
            for n in range(1, N - 1):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N - 1) * n * k)
            return res
        if t == 2:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * k)
            return res
        if t == 3:
            res = np.zeros((N, N))
            res[:, 0] = 0.5
            for n in range(1, N):
                for k in range(N):
                    res[k, n] = np.cos(np.pi / (N) * n * (k + 0.5))
            return res
        if t == 4:
            res = np.zeros((N, N))
            for k in range(N):
                for n in range(N):
                    res[k, n] = np.cos(np.pi / (N) * (n + 0.5) * (k + 0.5))
            return res

    def filter_from_matrix(self, i, j, size):
        mat = self.dct_matrix(t=self.type, N=size)
        fltr = (
            mat[i, : self.kernel_size]
            .reshape((-1, 1))
            .dot(mat[j, : self.kernel_size].reshape(1, -1))
        )
        return torch.as_tensor(fltr)

    def forward(self, x):
        in_ = x
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )  # int(self.K/
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.noise is not None:
            x.add_(self.noise.unsqueeze(-1).unsqueeze(-1))
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
        x = F.relu(x)
        return x


class HadamardBlock(TransformBlock):
    def __init__(self, *args, walsh=False, droprate=0.5, **kwargs):
        """
        Block based on Hadamard transform. Two types of transform are usable:
        Hadamard-Walsh transform and Hadamard-Paley transform.
        """
        super(HadamardBlock, self).__init__(*args, **kwargs)
        self.walsh = walsh
        self.filter_bank = self.get_filter_bank(
            kernel_size=self.kernel_size,  # kernel size
            input_channels=self.input_channels,
            walsh=self.walsh,  # type of Hadamard Transform
            lmbda=self.lmbda,
            diag=self.diag,
        ).float()
        if self._cuda:
            # Checks if cuda is available at all
            self.filter_bank = self.filter_bank.cuda()
        self.conv = nn.Conv2d(
            in_channels=self.filter_bank.shape[0],
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=self.bias,
        )
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(droprate)

    def filter_from_matrix(self, i, j, size):
        def paley(n):
            N = 2 ** n
            P_1 = np.array([1])
            P_2 = np.block([[np.kron(P_1, [1, 1])], [np.kron(P_1, [1, -1])]])
            if N == 1:  # n=0
                return P_1
            elif N == 2:  # n=1
                return P_2
            else:
                i = 2
                while i >= 2 and i <= n:
                    P_1 = P_2
                    P_2 = np.block(
                        [[np.kron(P_1, [1, 1])], [np.kron(P_1, [1, -1])]]
                    )

                    i += 1

            return P_2

        if self.walsh:
            h = hadamard(
                min(2 ** (size - 1).bit_length(), 1024)
            )  # /np.sqrt(32)
        else:
            h = paley(size)
        f = np.dot(h[i, :size].reshape(-1, 1), h[j, :size].reshape(1, -1))
        return torch.as_tensor(f)

    def forward(self, x):
        in_ = x
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )  # int(self.K/
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.noise is not None:
            x.add_(self.noise.unsqueeze(-1).unsqueeze(-1))
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
        x = F.relu(x)  # ?? do we need this relu layer?
        return x


class SlantBlock(TransformBlock):
    def __init__(self, *args, droprate=0.5, **kwargs):
        super(SlantBlock, self).__init__(*args, **kwargs)
        self.filter_bank = self.get_filter_bank(
            kernel_size=self.kernel_size,  # kernel size
            input_channels=self.input_channels,
            lmbda=self.lmbda,
            diag=self.diag,
        ).float()
        if self._cuda:
            self.filter_bank = self.filter_bank.cuda()
        self.conv = nn.Conv2d(
            in_channels=self.filter_bank.shape[0],
            out_channels=self.output_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=self.bias,
        )
        if self.bn:
            self.bnorm = nn.BatchNorm2d(self.filter_bank.shape[0])
        if self.drop:
            self.dropout = nn.Dropout(droprate)

    def filter_from_matrix(self, i, j, size):
        def slant(n):
            N = (2 ** (size - 1)).bit_length()
            S_1 = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

            an = np.sqrt((2 * N ** 2) / (4 * N ** 2 - 1))  # a2
            bn = np.sqrt((N ** 2 - 1) / (4 * N ** 2 - 1))  # b2

            S_2 = (
                1
                / np.sqrt(2)
                * np.array(
                    [
                        [1, 0, 1, 0],
                        [an, bn, -an, bn],
                        [0, 1, 0, -1],
                        [-bn, an, bn, an],
                    ]
                )
            )

            S_2 = np.matmul(S_2, block_diag(S_1, S_1))

            if N == 2:
                return S_1
            elif N == 4:
                return S_2
            else:
                S_prev = S_2
                i = 3
                while i >= 3 and i <= n:
                    N = 2 ** i
                    an = np.sqrt((3 * N ** 2) / (4 * N ** 2 - 1))  # a2
                    bn = np.sqrt((N ** 2 - 1) / (4 * N ** 2 - 1))
                    An1 = np.array([[1, 0], [an, bn]])
                    An2 = np.array([[1, 0], [-an, bn]])
                    Bn1 = np.array([[0, 1], [-bn, an]])
                    Bn2 = np.array([[0, -1], [bn, an]])
                    S_N = np.block(
                        [
                            [
                                An1,
                                np.zeros((2, N // 2 - 2)),
                                An2,
                                np.zeros((2, N // 2 - 2)),
                            ],
                            [
                                np.zeros((N // 2 - 2, 2)),
                                np.eye(N // 2 - 2),
                                np.zeros((N // 2 - 2, 2)),
                                np.eye(N // 2 - 2),
                            ],
                            [
                                Bn1,
                                np.zeros((2, N // 2 - 2)),
                                Bn2,
                                np.zeros((2, N // 2 - 2)),
                            ],
                            [
                                np.zeros((N // 2 - 2, 2)),
                                np.eye(N // 2 - 2),
                                np.zeros((N // 2 - 2, 2)),
                                -np.eye(N // 2 - 2),
                            ],
                        ]
                    )

                    S_N = (
                        1
                        / np.sqrt(2)
                        * np.matmul(S_N, block_diag(S_prev, S_prev))
                    )
                    S_prev = S_N
                    i += 1
                return S_prev

        s = slant(min(size, 8))  # /np.sqrt(32)
        f = np.dot(s[i, :size].reshape(-1, 1), s[j, :size].reshape(1, -1))
        return torch.as_tensor(f)

    def forward(self, x):
        in_ = x
        x = F.conv2d(
            x.float(),
            weight=self.filter_bank,
            padding=self.padding,
            stride=self.stride,
            groups=self.input_channels,
        )  # int(self.K/
        x = self.alpha_rooting(x, alpha=self.alpha_root)
        if self.noise is not None:
            x.add_(self.noise.unsqueeze(-1).unsqueeze(-1))
        if self.bn:
            x = F.relu(self.bnorm(x))
        else:
            x = F.relu(x)
        if self.drop:
            x = self.dropout(x)
        if self.use_res:
            x = self.conv(x) + self.shortcut(in_)
        else:
            x = self.conv(x)
        x = F.relu(x)  # ?? do we need this relu layer?
        return x


class ResNextTransofrmBlock(nn.Module):
    expansion = 2

    def __init__(
        self,
        input_channels,
        cardinality,
        transform_conv,
        bottleneck_width=4,
        stride=1,
    ):
        super(ResNextTransofrmBlock, self).__init__()

        self.conv0 = transform_conv(
            input_channels, kernel_size=3, padding=1, stride=1, lmbda=2
        )
        self.bn0 = nn.BatchNorm2d(self.conv0.filter_bank.size(0))
        group_width = cardinality * bottleneck_width
        self.linear = nn.Conv2d(
            in_channels=self.conv0.filter_bank.shape[0],
            out_channels=group_width,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(
            group_width,
            group_width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(
            group_width,
            self.expansion * group_width,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    self.expansion * group_width,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * group_width),
            )

    def forward(self, x):
        # transform -> normalize -> activate -> linear combination
        out = self.linear(F.relu(self.bn0(self.conv0(x))))
        out = F.relu(self.bn1(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


def test():
    return


if __name__ == "__main__":
    test()
