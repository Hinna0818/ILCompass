"""v3 CLI: reuse the immutable v1 trainer, optimizer, MLM and checkpoint protocol."""
from pathlib import Path
import sys

import train as core
from model_v3 import V3Config, V3Transformer

DEFAULT_CONFIG = core.ROOT / 'scratch/configs/training_v3.json'
_base_frozen_inputs = core.frozen_inputs


def frozen_inputs(input_config):
    prepared, provenance, assets = _base_frozen_inputs(input_config)
    # Add v3 code without changing hashes embedded in historical v1/v2 checkpoints.
    for path in [Path(__file__), Path(__file__).with_name('model_v3.py'), Path(__file__).with_name('prepare_v3.py')]:
        provenance[str(path.relative_to(core.ROOT))] = core.file_hash(path)
    return prepared, provenance, assets


def main():
    core.__doc__ = __doc__
    core.ModelConfig, core.ILC2Transformer = V3Config, V3Transformer
    core.frozen_inputs = frozen_inputs
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', str(DEFAULT_CONFIG)])
    core.main()


if __name__ == '__main__':
    main()
