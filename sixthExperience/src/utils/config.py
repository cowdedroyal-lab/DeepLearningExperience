import os
from typing import Any, Dict
import yaml

class Config:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        for k, v in data.items():
            setattr(self, k, v)
        # 类型规范化，确保数值/布尔为正确类型，增强鲁棒性
        self._normalize_types()
        self._validate()

    @staticmethod
    def _to_float(v, name: str):
        if isinstance(v, bool):
            raise ValueError(f"字段 {name} 期望为浮点数，实际为布尔：{v}")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"字段 {name} 无法转换为浮点数：{v}")
        raise ValueError(f"字段 {name} 类型不被支持：{type(v)}")

    @staticmethod
    def _to_int(v, name: str):
        if isinstance(v, bool):
            # 明确拒绝将布尔当作整型（True->1/False->0），避免隐式错误
            raise ValueError(f"字段 {name} 期望为整数，实际为布尔：{v}")
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            try:
                # 允许如 "8" 或 "8.0" 或科学计数法
                fv = float(v)
                return int(fv)
            except ValueError:
                raise ValueError(f"字段 {name} 无法转换为整数：{v}")
        raise ValueError(f"字段 {name} 类型不被支持：{type(v)}")

    @staticmethod
    def _to_bool(v, name: str):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "yes", "y", "on"): return True
            if s in ("false", "0", "no", "n", "off"): return False
            raise ValueError(f"字段 {name} 无法转换为布尔：{v}")
        if isinstance(v, (int, float)):
            return bool(v)
        raise ValueError(f"字段 {name} 类型不被支持：{type(v)}")

    def _normalize_types(self):
        # 数值型
        if hasattr(self, 'num_classes'):
            self.num_classes = self._to_int(self.num_classes, 'num_classes')
        if hasattr(self, 'ignore_index'):
            self.ignore_index = self._to_int(self.ignore_index, 'ignore_index')
        if hasattr(self, 'input_height'):
            self.input_height = self._to_int(self.input_height, 'input_height')
        if hasattr(self, 'input_width'):
            self.input_width = self._to_int(self.input_width, 'input_width')
        if hasattr(self, 'batch_size'):
            self.batch_size = self._to_int(self.batch_size, 'batch_size')
        if hasattr(self, 'epochs'):
            self.epochs = self._to_int(self.epochs, 'epochs')
        if hasattr(self, 'num_workers'):
            self.num_workers = self._to_int(self.num_workers, 'num_workers')
        if hasattr(self, 'val_interval'):
            self.val_interval = self._to_int(self.val_interval, 'val_interval')
        if hasattr(self, 'log_interval'):
            self.log_interval = self._to_int(self.log_interval, 'log_interval')
        if hasattr(self, 'warmup_epochs'):
            self.warmup_epochs = self._to_int(self.warmup_epochs, 'warmup_epochs')

        if hasattr(self, 'lr'):
            self.lr = self._to_float(self.lr, 'lr')
        if hasattr(self, 'weight_decay'):
            self.weight_decay = self._to_float(self.weight_decay, 'weight_decay')

        # 布尔型
        if hasattr(self, 'amp'):
            self.amp = self._to_bool(self.amp, 'amp')

        # 字符串规格化
        if hasattr(self, 'optimizer') and isinstance(self.optimizer, str):
            self.optimizer = self.optimizer.strip()
        if hasattr(self, 'lr_scheduler') and isinstance(self.lr_scheduler, str):
            self.lr_scheduler = self.lr_scheduler.strip()

    def _validate(self):
        required = [
            'experiment_name','num_classes','ignore_index','train_images_dir','train_labels_dir',
            'val_images_dir','val_labels_dir','input_height','input_width','batch_size','epochs','lr'
        ]
        missing = [r for r in required if r not in self.data]
        if missing:
            raise ValueError(f'配置缺失字段: {missing}')
        # 数据路径存在性强校验（更稳健）
        if not os.path.exists(self.train_images_dir):
            raise FileNotFoundError(f"训练图像目录不存在: {self.train_images_dir}")
        if not os.path.exists(self.train_labels_dir):
            raise FileNotFoundError(f"训练标签目录不存在: {self.train_labels_dir}")
        if not os.path.exists(self.val_images_dir):
            raise FileNotFoundError(f"验证图像目录不存在: {self.val_images_dir}")
        if not os.path.exists(self.val_labels_dir):
            raise FileNotFoundError(f"验证标签目录不存在: {self.val_labels_dir}")
        if self.num_classes <= 1:
            raise ValueError('num_classes 必须 > 1')

    @property
    def image_size(self):
        return (self.input_height, self.input_width)

def load_config(path: str) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(data)

if __name__ == '__main__':
    cfg = load_config('../../configs/camvid.yaml')
    print(cfg.experiment_name, cfg.image_size)
