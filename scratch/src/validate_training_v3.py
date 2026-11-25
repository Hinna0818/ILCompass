"""Reuse real-batch forward/backward, PAD and exact-resume acceptance for v3."""
import json
import sys

import validate_training as acceptance
from model_v3 import V3Config, V3Transformer
from train_v3 import DEFAULT_CONFIG, frozen_inputs


if __name__ == '__main__':
    acceptance.__doc__ = __doc__
    acceptance.ModelConfig, acceptance.ILC2Transformer = V3Config, V3Transformer
    acceptance.frozen_inputs = frozen_inputs
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', str(DEFAULT_CONFIG)])
    acceptance.main()
    config_path = sys.argv[sys.argv.index('--config') + 1]
    cfg = json.loads(acceptance.Path(config_path).read_text())
    out = (acceptance.ROOT / cfg['training_acceptance']).parent
    gate = json.loads((out / 'validation.json').read_text())
    maximum = max(gate['measurements'], key=lambda row: row['peak_allocated_GiB'])
    text = f'''# Scratch v3 MLM-only：真实GPU验收

模型参数：{gate['parameters']:,}；8层384维Pre-LN，只有原始0.2 ID CE＋0.8 Value MSE。
正式预训练未启动；本验收的随机诊断权重已丢弃。

- GPU：{gate['gpu']}
- 全部真实最长输入：{gate['real_max_length']:,} Token
- 测试case最高allocated：{maximum['peak_allocated_GiB']:.3f} GiB
- 该case reserved：{maximum['peak_reserved_GiB']:.3f} GiB
- 断点下一次更新最大绝对误差：{gate['checkpoint_next_update_max_absolute_error']:.8g}
- 完整encoder PAD不变性、真实双worker与梯度累积均通过。
- 实际SDPA操作：{', '.join(gate['actual_sdpa_operations'])}

逐case时长与峰值见gpu_benchmark.csv，仅为短诊断，不是完整epoch训练速度。
训练启动：从scratch目录执行 `bash src/run_v3.sh`；参数在configs/training_v3.json。
'''
    (out / 'report.md').write_text(text)
