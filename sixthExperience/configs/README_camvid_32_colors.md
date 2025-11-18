# CamVid 32 类颜色映射示例

当你的 CamVid 标签为彩色 PNG（每种颜色表示一个类别）时，需要提供一个“颜色 -> 类别 id”的映射文件，供数据集将 RGB 标签转换为整数索引。

两种受支持的 JSON 格式：

1. 列表形式
```json
[
  {"color": [128, 128, 128], "id": 0},
  {"color": [128, 0, 0],     "id": 1}
]
```

2. 字典形式（键是 "r,g,b" 字符串）
```json
{
  "128,128,128": 0,
  "128,0,0": 1
}
```

使用方式：
- 将完整 32 类映射填入 `configs/camvid_32_colors.json`（建议从数据集官方文档或权威开源实现复制颜色与类别编号）。
- 在配置 `configs/camvid.yaml` 中添加字段：

```yaml
num_classes: 32
class_mode: camvid_32
class_colors_path: ./configs/camvid_32_colors.json
```

注意：
- 如果你的标签 PNG 是索引图（P/L 模式），像素值就是类别 id（0..31），无需颜色映射文件，`class_colors_path` 可省略。
- 若标签为 RGB 且不提供映射，程序会报错提示。
