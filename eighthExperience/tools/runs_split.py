import os
import shutil

def split_events(
    runs_dir: str = "runs",
    source_folder: str = "coco_baseline",
    target_prefix: str = "log_"
):
    source_path = os.path.join(runs_dir, source_folder)
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Source folder not found: {source_path}")

    # 找到所有 event* 文件
    event_files = [
        f for f in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, f)) and f.startswith("event")
    ]

    if not event_files:
        print("No event* files found.")
        return

    # 按顺序放到 log_1, log_2, ...
    for idx, event_file in enumerate(sorted(event_files), start=1):
        log_dir_name = f"{target_prefix}{idx}"
        log_dir = os.path.join(runs_dir, log_dir_name)

        # 创建 log_i 目录
        os.makedirs(log_dir, exist_ok=True)

        src = os.path.join(source_path, event_file)
        dst = os.path.join(log_dir, event_file)

        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

    print("Done.")

if __name__ == "__main__":
    # 当前脚本位置假设在 tools 目录下
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(project_root, "runs")

    split_events(runs_dir=runs_dir, source_folder="coco_baseline", target_prefix="log_")