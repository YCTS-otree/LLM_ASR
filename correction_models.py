"""Lifecycle of selected local models; each retains its own worker and transcript."""
from dataclasses import replace
from qwen_backend import QwenLocalBackend
from qwen_spec import MODEL_SPECS


class CorrectionModels:
    source = 'LOCAL_LLM'

    def __init__(self, config, status=lambda state: None, mode=None, comparison_dtype=None):
        self.mode = mode or config.model_size
        sizes = ('2b', '0.8b') if self.mode == 'both' else (self.mode,)
        self.states = {size: 'Loading' for size in sizes}
        self.status = status
        self.members = tuple(QwenLocalBackend(
            replace(config, model_size=size,
                    dtype=comparison_dtype if self.mode=='both' and size=='0.8b' and comparison_dtype else config.dtype,
                    model_path=config.model_path if size == config.model_size else str(MODEL_SPECS[size]['path'])),
            lambda state, size=size: self.changed(size, state)) for size in sizes)
        self.config = self.members[0].config
        self.cancel = _CancelAll(self)

    def changed(self, size, state):
        self.states[size] = state
        self.status('  |  '.join(f'Qwen3.5-{s.upper()} · {v}' for s,v in self.states.items()))

    @property
    def ready(self):
        return all(member.ready for member in self.members)

    @property
    def enabled(self):
        return any(member.enabled for member in self.members)

    @property
    def load_count(self):
        return sum(member.load_count for member in self.members)

    def set_enabled(self, enabled):
        for member in self.members:
            member.set_enabled(enabled)

    def load(self):
        errors = []
        for member in self.members:
            try:
                member.load()
            except Exception as exc:
                errors.append(f'{member.config.model_size}: {exc}')
        if errors:
            raise RuntimeError('; '.join(errors))

    def process(self, context):
        return self.members[0].process(context)

    def close(self):
        self.cancel.set()
        for member in self.members:
            member.close()


class _CancelAll:
    def __init__(self, owner):
        self.owner = owner

    def set(self):
        for member in self.owner.members:
            member.cancel.set()
