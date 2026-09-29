"""Backend-neutral evidence. Times are seconds; offsets count captured samples."""
from dataclasses import dataclass, field, asdict
from typing import Any
import uuid


END_REASONS = {'silence', 'max_duration', 'manual_stop', 'shutdown_flush', 'end_of_file'}


@dataclass(frozen=True)
class AudioSegment:
    segment_id: str
    start_sample: int
    end_sample: int
    sample_rate: int
    end_reason: str
    audio: Any = field(repr=False, compare=False)
    audio_rate: int = 16000
    pause_after_ms: float | None = None

    def __post_init__(self):
        if self.end_reason not in END_REASONS or self.sample_rate <= 0:
            raise ValueError('Invalid audio segment')
        if not 0 <= self.start_sample <= self.end_sample:
            raise ValueError('Invalid sample range')

    @property
    def start_time(self):
        return self.start_sample / self.sample_rate

    @property
    def end_time(self):
        return self.end_sample / self.sample_rate

    @property
    def duration(self):
        return self.end_time - self.start_time

    def __len__(self):
        return len(self.audio)

    @classmethod
    def from_audio(cls, audio, rate=16000, reason='manual_stop'):
        return cls(uuid.uuid4().hex, 0, len(audio), rate, reason, audio, rate)


@dataclass(frozen=True)
class Hypothesis:
    rank: int
    text: str
    score: float | None


@dataclass(frozen=True)
class ASREvidence:
    segment_id: str
    start_time: float
    end_time: float
    best_text: str
    nbest: tuple[Hypothesis, ...] = ()
    timestamps: tuple[tuple[float, float], ...] = ()
    raw_result: Any = None
    end_reason: str = 'silence'
    confidence: None = None
    start_sample: int = 0
    end_sample: int = 0
    sample_rate: int = 16000
    inference_latency: float = 0.0
    beam_size: int = 1
    requested_nbest: int = 1
    timestamp_warning: str | None = None
    pause_after_ms: float | None = None
    inference_precision: str | None = None
    fallback_reason: str | None = None
    punctuated_text: str | None = None
    punctuation_error: str | None = None
    low_energy_spans: tuple[tuple[float, float], ...] = ()
    pause_threshold: float | None = None

    @property
    def global_timestamps(self):
        return tuple((self.start_time + a, self.start_time + b) for a, b in self.timestamps)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StableASREvidence:
    evidence: ASREvidence
    previous_segment_id: str | None
    hard_cut: bool
    follows_hard_cut: bool
    # A pause is an endpoint signal, not proof of grammatical sentence completion.
    natural_pause: bool


class StabilityBuffer:
    """Serial offline ASR emits one final observation per segment; never adds punctuation."""
    def __init__(self):
        self.previous = None

    def push(self, evidence):
        previous = self.previous
        if previous and (evidence.segment_id == previous.segment_id or
                         evidence.start_time < previous.end_time - 1e-6):
            raise ValueError('Duplicate or out-of-order ASR segment')
        result = StableASREvidence(evidence, previous.segment_id if previous else None,
                                  evidence.end_reason == 'max_duration',
                                  bool(previous and previous.end_reason == 'max_duration'),
                                  evidence.end_reason == 'silence')
        self.previous = evidence
        return result
