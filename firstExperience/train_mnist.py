# -----------------------------------------------------------------------------
# 实验一：手写数字识别 - 完整代码实现
# -----------------------------------------------------------------------------
# 导入必要的库
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# -----------------------------------------------------------------------------
# 1. 参数管理模块 (argparse)
# -----------------------------------------------------------------------------
# 使用 argparse 统一管理所有可配置的变量和超参数
def get_args():
    """定义并解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于PyTorch的MNIST手写数字识别训练脚本")

    # -- 数据集和路径参数 --
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='MNIST数据集的存储目录 (默认: ./data)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='TensorBoard日志文件的存储目录 (默认: ./logs)')
    parser.add_argument('--save-path', type=str, default='./models/best_mnist_cnn.pth',
                        help='最佳模型权重的保存路径 (默认: ./models/best_mnist_cnn.pth)')

    # -- 模型训练超参数 --
    parser.add_argument('--batch-size', type=int, default=64,
                        help='训练和测试时每个批次的大小 (默认: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练的总轮数 (默认: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='优化器的学习率 (默认: 0.001)')

    # -- 系统配置参数 --
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='强制使用CPU进行训练')

    return parser.parse_args()


# -----------------------------------------------------------------------------
# 2. 网络结构设计模块
# -----------------------------------------------------------------------------
# 构建一个规范的卷积神经网络 (CNN)
class CNN(nn.Module):
    """
    卷积神经网络模型结构
    流程: (卷积 -> 激活 -> 池化) x 2 -> 展平 -> 全连接
    """
    def __init__(self):
        super(CNN, self).__init__()

        # 第一个卷积模块: 提取基础边缘特征
        # 输入: (batch_size, 1, 28, 28)
        self.conv1 = nn.Sequential(
            # 卷积层: 1个输入通道, 16个输出通道, 5x5卷积核, 步长1, 填充2
            # padding = (kernel_size - 1) / 2 = (5-1)/2 = 2, 保持图像尺寸不变
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            # 激活函数: 引入非线性
            nn.ReLU(),
            # 池化层: 2x2最大池化, 进行下采样, 提取显著特征
            # 输出尺寸: (28 / 2) = 14 -> (batch_size, 16, 14, 14)
            nn.MaxPool2d(kernel_size=2),
        )

        # 第二个卷积模块: 提取更复杂的组合特征
        # 输入: (batch_size, 16, 14, 14)
        self.conv2 = nn.Sequential(
            # 卷积层: 16个输入通道, 32个输出通道, 5x5卷积核, 步长1, 填充2
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            # 池化层: 2x2最大池化
            # 输出尺寸: (14 / 2) = 7 -> (batch_size, 32, 7, 7)
            nn.MaxPool2d(2),
        )

        # 全连接层 (分类器)
        # 将池化后的多维特征图展平后，送入全连接层进行分类
        # 输入维度: 32 (通道数) * 7 (高) * 7 (宽) = 1568
        # 输出维度: 10 (对应0-9十个类别)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """定义模型的前向传播路径"""
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平操作: 将 (batch_size, 32, 7, 7) 的张量展平为 (batch_size, 1568)
        # x.size(0) 是 batch_size
        x = x.view(x.size(0), -1)
        # 全连接层输出
        output = self.out(x)
        return output

# -----------------------------------------------------------------------------
# 3. 主执行函数模块
# -----------------------------------------------------------------------------
def main():
    # 获取所有配置参数
    args = get_args()

    # -- 1. 环境与设备配置 --
    # 检查CUDA是否可用，并据此设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"正在使用的设备: {device}")

    # -- 2. TensorBoard 设置 --
    # 创建日志文件夹
    os.makedirs(args.log_dir, exist_ok=True)
    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard 日志将保存在: {args.log_dir}")


    # -- 3. 数据集准备与加载 --
    # 定义数据预处理流程：将PIL Image或numpy.ndarray转换为Tensor，并归一化到[0, 1]
    transform = transforms.ToTensor()

    # 下载并加载训练数据集
    # 如果 data_dir 中没有数据集, download=True 会自动下载
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    # 下载并加载测试数据集
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    # 创建数据加载器 (DataLoader)，用于分批次加载数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    print("MNIST 数据集加载完成。")


    # -- 4. 初始化模型、损失函数和优化器 --
    # 实例化CNN模型并将其移动到指定设备
    model = CNN().to(device)

    # 将模型结构写入TensorBoard
    # 创建一个虚拟输入以获取图形
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, dummy_input)

    # 定义损失函数: 交叉熵损失，适用于多分类问题
    loss_func = nn.CrossEntropyLoss()

    # 定义优化器: Adam优化器，自适应调整学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # -- 5. 训练与评估循环 --
    print("======== 开始训练 ========")
    best_accuracy = 0.0  # 用于记录最佳准确率

    for epoch in range(args.epochs):
        # --- 训练阶段 ---
        model.train()  # 设置模型为训练模式
        train_loss_sum = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # 将数据和标签移动到指定设备
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 前向传播
            output = model(batch_x)
            # 计算损失
            loss = loss_func(output, batch_y)
            train_loss_sum += loss.item()

            # 反向传播与优化
            optimizer.zero_grad()  # 清除旧的梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新权重

        # 计算并记录平均训练损失
        avg_train_loss = train_loss_sum / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # --- 评估阶段 ---
        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 在此块内，所有计算都不会影响梯度
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device), test_y.to(device)
                outputs = model(test_x)
                # 获取预测结果: torch.max返回(值, 索引)，我们取索引[1]
                _, predicted = torch.max(outputs.data, 1)
                total += test_y.size(0)
                correct += (predicted == test_y).sum().item()

        # 计算并记录测试准确率
        accuracy = 100 * correct / total
        writer.add_scalar('Accuracy/test', accuracy, epoch)

        # 打印当前轮次的训练信息
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Accuracy: {accuracy:.2f}%")

        # -- 6. 保存最佳模型 --
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 确保保存路径的目录存在
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            # 保存模型的状态字典
            torch.save(model.state_dict(), args.save_path)
            print(f"发现新的最佳模型! 准确率: {best_accuracy:.2f}%, 模型已保存至 {args.save_path}")


    # -- 7. 结束工作 --
    writer.close()
    print("======== 训练完成 ========")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    print(f"模型权重已保存在: {args.save_path}")
    print(f"运行 'tensorboard --logdir={args.log_dir}' 来查看训练过程。")


# -----------------------------------------------------------------------------
# 程序入口
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()