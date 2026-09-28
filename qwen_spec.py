"""Pinned official model identity; no runtime imports or network access."""
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3.5-2B'
MODEL_REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'Qwen3.5-2B'
WEIGHT_FILE = 'model.safetensors-00001-of-00001.safetensors'
WEIGHT_SHA256 = 'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1'

MODEL_SPECS = {
    '2b': dict(model_id=MODEL_ID, revision=MODEL_REVISION, path=DEFAULT_MODEL_PATH,
               sha256=WEIGHT_SHA256),
    '0.8b': dict(model_id='Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17',
                path=DEFAULT_MODEL_PATH.parent / 'Qwen3.5-0.8B',
                sha256='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'),
}
