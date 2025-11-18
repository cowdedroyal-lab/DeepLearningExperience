import torch
import torch.nn as nn
from typing import List, Tuple, Optional

try:
    # torchvision 可能并未安装到环境中，这里做一次安全导入
    from torchvision.models import vgg16_bn
except Exception:  # pragma: no cover - 可运行但在无torchvision环境下退化
    vgg16_bn = None

# SegNet 编码器-解码器：使用 MaxPool2d(return_indices=True) + MaxUnpool2d
# 结构：多个卷积块；每个块含 2 个 (Conv+BN+ReLU) 后接池化(encoder)，解码阶段逆序使用对应的 indices。

class ConvBlock(nn.Module):
    """基础卷积块：Conv-BN-ReLU * 2

    该实现与 VGG16-BN 单个 block 结构兼容，便于后续加载预训练权重。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SegNet(nn.Module):
    """SegNet 语义分割网络

    - 支持两种变体：
      - ``variant='vanilla'``: 保持当前简化版结构（兼容已有实验）。
      - ``variant='vgg16_bn'``: 使用完整 VGG16-BN 编码器(2-2-3-3-3 卷积层)，并可加载 ImageNet 预训练权重。
    - 通过 MaxPool(return_indices=True) 与 MaxUnpool 传递池化索引。
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 11,
                 encoder_channels: Optional[List[int]] = None,
                 variant: str = 'vanilla',
                 pretrained: bool = False,
                 freeze_encoder_bn: bool = False) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.variant = variant

        if encoder_channels is None:
            # 简化版与 VGG16-GAP 通道数保持一致
            encoder_channels = [64, 128, 256, 512, 512]
        self.encoder_channels = encoder_channels

        # -----------------
        # 构建编码器 / 解码器
        # -----------------
        if variant not in {"vanilla", "vgg16_bn"}:
            raise ValueError(f"未知 SegNet variant: {variant}")

        # 对两种变体使用统一的 block 列表与池化层列表，以保证前向逻辑简单
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.ModuleList()

        self._build_encoder()
        self._build_decoder()

        self.classifier = nn.Conv2d(self.encoder_channels[0], num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')

        # 预训练权重加载放在网络构建之后，避免 Decoder 权重被预训练覆盖
        if self.variant == 'vgg16_bn' and pretrained:
            self._load_vgg16_bn_pretrained()

        if freeze_encoder_bn:
            self.freeze_bn()

    # ------------------------------------------------------------------
    # 构建编码器 / 解码器
    # ------------------------------------------------------------------
    def _build_encoder(self) -> None:
        """根据 variant 构建编码器卷积块与池化层。

        对于 ``vgg16_bn`` 变体，block 里卷积层数量为:
        - Block1: 2 个 conv (64)
        - Block2: 2 个 conv (128)
        - Block3: 3 个 conv (256)
        - Block4: 3 个 conv (512)
        - Block5: 3 个 conv (512)
        这里通过显式构建卷积列表来对齐 VGG16-BN 的层顺序，以利于权重映射。
        """

        in_ch = self.in_channels
        channels = self.encoder_channels

        if self.variant == 'vanilla':
            # 与旧实现保持完全一致：每个 block 2 个 conv
            for ch in channels:
                block = ConvBlock(in_ch, ch)
                self.enc_blocks.append(block)
                self.pool.append(nn.MaxPool2d(2, stride=2, return_indices=True))
                in_ch = ch
        else:  # vgg16_bn
            # VGG16-BN conv 层个数配置
            num_convs_per_block = [2, 2, 3, 3, 3]
            assert len(channels) == len(num_convs_per_block), "encoder_channels 必须与 VGG16 block 数一致"

            blocks: List[nn.Sequential] = []
            for ch, n_convs in zip(channels, num_convs_per_block):
                layers: List[nn.Module] = []
                for _ in range(n_convs):
                    conv = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False)
                    bn = nn.BatchNorm2d(ch)
                    relu = nn.ReLU(inplace=True)
                    layers.extend([conv, bn, relu])
                    in_ch = ch
                block = nn.Sequential(*layers)
                blocks.append(block)
                self.pool.append(nn.MaxPool2d(2, stride=2, return_indices=True))

            self.enc_blocks = nn.ModuleList(blocks)

    def _build_decoder(self) -> None:
        # 解码器与编码器对称，使用 ConvBlock 或顺序卷积以恢复空间分辨率
        dec_blocks = []
        decoder_channels = list(reversed(self.encoder_channels))
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1] if i < len(decoder_channels) - 1 else self.encoder_channels[0]
            dec_blocks.append(ConvBlock(in_ch, out_ch))
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.unpool = nn.ModuleList([nn.MaxUnpool2d(2, stride=2) for _ in decoder_channels])

    # ------------------------------------------------------------------
    # 预训练权重加载
    # ------------------------------------------------------------------
    def _load_vgg16_bn_pretrained(self) -> None:
        """从 torchvision.vgg16_bn 加载预训练 encoder 权重。

        仅在 ``variant='vgg16_bn'`` 时生效。若 torchvision 不可用，将给出友好提示但不中断训练。
        """

        if vgg16_bn is None:
            print("[SegNet] torchvision 未安装，无法加载 VGG16-BN 预训练权重，继续使用随机初始化编码器")
            return

        try:
            backbone = vgg16_bn(weights="IMAGENET1K_V1")  # torch>=1.12 写法；旧版可改为 pretrained=True
        except TypeError:
            backbone = vgg16_bn(pretrained=True)  # 兼容旧版本 torchvision

        # VGG16-BN 的 features 是一串 conv-bn-relu + pool，我们需要将其按 block 切片
        vgg_features = list(backbone.features.children())

        # 每个 block 的 conv-bn-relu 层数 (不含 MaxPool)
        num_layers_per_block = [6, 6, 9, 9, 9]  # (conv+bn+relu)*[2,2,3,3,3]
        blocks_slices = []
        start = 0
        for n in num_layers_per_block:
            end = start + n
            blocks_slices.append(vgg_features[start:end])
            # 跳过紧随其后的 MaxPool 层
            start = end + 1

        # 将权重逐 block 映射到 self.enc_blocks
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for seg_block, vgg_block_layers in zip(self.enc_blocks, blocks_slices):
            for seg_layer, vgg_layer in zip(seg_block.children(), vgg_block_layers):
                if isinstance(seg_layer, nn.Conv2d) and isinstance(vgg_layer, nn.Conv2d):
                    key = None
                    for name, param in own_state.items():
                        if param.data_ptr() == seg_layer.weight.data_ptr():
                            key = name
                            break
                    if key is not None:
                        own_state[key].copy_(vgg_layer.weight.data)
                        loaded += 1
                elif isinstance(seg_layer, nn.BatchNorm2d) and isinstance(vgg_layer, nn.BatchNorm2d):
                    w_key = b_key = None
                    for name, param in own_state.items():
                        if param.data_ptr() == seg_layer.weight.data_ptr():
                            w_key = name
                        elif param.data_ptr() == seg_layer.bias.data_ptr():
                            b_key = name
                    if w_key is not None:
                        own_state[w_key].copy_(vgg_layer.weight.data)
                    if b_key is not None:
                        own_state[b_key].copy_(vgg_layer.bias.data)
                    loaded += 1
                else:
                    skipped += 1

        print(f"[SegNet] 已从 VGG16-BN 预训练模型加载 {loaded} 个编码器参数，跳过 {skipped} 个层(结构不匹配或非 Conv/BN)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_list: List[torch.Tensor] = []
        sizes_list: List[Tuple[int, int, int, int]] = []

        # Encoder
        for block, pool in zip(self.enc_blocks, self.pool):
            x = block(x)
            sizes_list.append(x.size())
            x, indices = pool(x)
            indices_list.append(indices)

        # Decoder (逆序使用 indices)
        for i, (block, unpool) in enumerate(zip(self.dec_blocks, self.unpool)):
            # 对应 encoder 的倒序层尺寸
            enc_size = sizes_list[-(i+1)]
            enc_indices = indices_list[-(i+1)]
            x = unpool(x, enc_indices, output_size=enc_size)
            x = block(x)

        x = self.classifier(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

if __name__ == '__main__':
    # 快速自测
    model = SegNet(in_channels=3, num_classes=11)
    inp = torch.randn(2, 3, 360, 480)
    with torch.no_grad():
        out = model(inp)
    print('输入:', inp.shape, '输出:', out.shape)  # 期望 (2,11,360,480)
