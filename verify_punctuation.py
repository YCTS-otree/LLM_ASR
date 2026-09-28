"""Actual same-model v1.2.0/v1.2.1 prompt comparison; no microphone/network."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import qwen_backend
from qwen_backend import QwenLocalBackend
from llm_protocol import build_messages, PROMPT_VERSION
from llm_pipeline import ContextBuilder
from evidence import ASREvidence, Hypothesis, StabilityBuffer
from transcript_store import TranscriptStore
from patches import RevisionEngine, PatchRejected
from settings import LocalLLMConfig
from gpu_telemetry import GPUSampler
from punctuation_cases import CASES, wording, acceptable
from observability import configure_logging


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dtype',choices=['bf16','int4'],default='bf16')
    parser.add_argument('--current-only',action='store_true')
    parser.add_argument('--model',choices=['2b','0.8b'],default='2b')
    args=parser.parse_args()
    configure_logging(Path('logs'))
    from benchmarks.v120_prompt import build_messages as previous
    from qwen_spec import MODEL_SPECS
    backend=QwenLocalBackend(replace(LocalLLMConfig.from_environment(),dtype=args.dtype,
                                   model_size=args.model,model_path=str(MODEL_SPECS[args.model]['path'])))
    backend.load()
    sampler=GPUSampler().start()
    records=[]
    try:
        current_label=PROMPT_VERSION.rsplit('_',1)[-1]
        variants=[(current_label,build_messages)] if args.current_only else [('v5',previous),(current_label,build_messages)]
        for label,builder in variants:
            qwen_backend.build_messages=builder
            for name,segments,expected in CASES:
                store=TranscriptStore()
                stability=StabilityBuffer()
                for i,(text,reason) in enumerate(segments):
                    evidence=ASREvidence(f'{name}_{i}',i,i+1,text,(Hypothesis(1,text,-1.0),),end_reason=reason,
                                         pause_after_ms=800 if reason=='silence' else 0)
                    stable=stability.push(evidence)
                    store.append(stable)
                context=ContextBuilder().build(store,stable)
                start=time.perf_counter()
                result=backend.process(context)
                end=time.perf_counter()
                status=result.error
                if not status:
                    try:
                        RevisionEngine(store).apply(result.patch,context.snapshot,'LOCAL_LLM',result.metadata)
                        status='APPLIED' if store.revision!=context.snapshot.revision else 'NO_CHANGE'
                    except PatchRejected as exc: status=exc.code
                final=store.canonical_text
                record=dict(prompt=label,dtype=args.dtype,case=name,result=status,final=final,expected=expected,
                    passed=acceptable(name,final,expected) and (status in ('APPLIED','NO_CHANGE') or
                        (name=='P5a' and result.metadata.get('detail')=='HARD_CUT_TERMINATOR')),
                    wording_changed=wording(final)!=wording(expected),patch=result.patch,
                    metadata=dict(result.metadata,prompt_version='qwen_asr_correction_'+label),
                    telemetry=sampler.summarize(start,end),segment_texts=[s.current_text for s in store.segments])
                records.append(record)
                print(json.dumps({k:record[k] for k in ('prompt','case','passed','result','final')},ensure_ascii=True),flush=True)
                store.close()
        suffix='' if args.model=='2b' else '_0_8b'
        path=Path('logs')/f'punctuation_{args.dtype}_{PROMPT_VERSION}{suffix}.json'
        path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAVED',path,flush=True)
    finally:
        backend.close()
        sampler.stop()
        qwen_backend.build_messages=build_messages


if __name__=='__main__': main()
