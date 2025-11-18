import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars_from_event(event_path: str, tags: list[str]):
    ea = EventAccumulator(event_path)
    ea.Reload()
    data = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = (steps, values)
        except KeyError:
            pass
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./logs', help='TensorBoard 日志目录')
    parser.add_argument('--outdir', type=str, default='./figures', help='输出图片目录')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 找到所有事件文件
    pattern = os.path.join(args.logdir, '**', 'events.out.tfevents.*')
    event_files = sorted(glob.glob(pattern, recursive=True))
    if not event_files:
        print('未找到事件文件，请确认日志目录。')
        return

    # 取最近的一个事件文件
    event_path = event_files[-1]
    print(f'使用事件文件: {event_path}')

    tags = ['Loss/train', 'Accuracy/test']
    scalars = load_scalars_from_event(event_path, tags)

    # 绘制 Loss 曲线
    if 'Loss/train' in scalars:
        steps, values = scalars['Loss/train']
        plt.figure(figsize=(6,4))
        plt.plot(steps, values, label='Train Loss', color='#1f77b4')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(args.outdir, 'loss_curve.png')
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f'Saved {out_path}')

    # 绘制 Accuracy 曲线
    if 'Accuracy/test' in scalars:
        steps, values = scalars['Accuracy/test']
        plt.figure(figsize=(6,4))
        plt.plot(steps, values, label='Test Accuracy', color='#2ca02c')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy over Epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(args.outdir, 'accuracy_curve.png')
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
