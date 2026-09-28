"""Opt-in real model quality and telemetry checks. No microphone or network."""
import argparse
from dataclasses import replace
import gc
import json
import statistics
import time
from pathlib import Path
from evidence import ASREvidence, Hypothesis, StabilityBuffer
from transcript_store import TranscriptStore
from llm_pipeline import CorrectionContext
from patches import RevisionEngine, PatchRejected
from settings import LocalLLMConfig
from qwen_backend import QwenLocalBackend
from qwen_cases import CASES
from gpu_telemetry import GPUSampler
from observability import configure_logging
from engines import ROOT
from llm_protocol import PROMPT_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['relative','anchored','both'], default='both')
    parser.add_argument('--dtype', choices=['bf16','fp16','int4'], default='bf16')
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    configure_logging(ROOT / 'logs')
    sampler = GPUSampler().start()
    import torch
    def memory():
        torch.cuda.synchronize()
        return dict(allocated_mib=torch.cuda.memory_allocated()/2**20,
                    reserved_mib=torch.cuda.memory_reserved()/2**20,
                    free_mib=torch.cuda.mem_get_info()[0]/2**20)
    records = []
    start = time.perf_counter()
    time.sleep(2)
    baseline = sampler.summarize(start, time.perf_counter())
    backend = QwenLocalBackend(replace(LocalLLMConfig.from_environment(), dtype=args.dtype))
    load_start = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - load_start
    resident = memory()
    protocols = ('relative', 'anchored') if args.protocol == 'both' else (args.protocol,)
    try:
        for protocol in protocols:
            backend.config = replace(backend.config, protocol=protocol)
            for repeat in range(args.repeats):
                for case in CASES:
                    store = TranscriptStore()
                    evidence = ASREvidence(case['name'],0,1,case['text'],
                        tuple(Hypothesis(i+1, t, -1.5-i*.3) for i,t in enumerate(case['candidates'])))
                    stable = StabilityBuffer().push(evidence)
                    store.append(stable)
                    context = CorrectionContext(store.snapshot(), replace(evidence,best_text=case['best']))
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    result = backend.process(context)
                    end = time.perf_counter()
                    outcome = result.error
                    if not result.error:
                        try:
                            before = store.revision
                            RevisionEngine(store).apply(result.patch,context.snapshot,'LOCAL_LLM',result.metadata)
                            outcome = 'APPLIED' if store.revision != before else 'NO_CHANGE'
                        except PatchRejected as exc:
                            outcome = exc.code
                    current = store.canonical_text
                    expected = case['expected']
                    if case.get('punctuation_allowed'):
                        import re
                        clean = lambda text: re.sub(r'[，。！？、；：,.!?;:\s]', '', text)
                        passed = clean(current) == clean(expected)
                    else:
                        passed = current == expected
                    passed = passed and outcome in ('APPLIED', 'NO_CHANGE')
                    record = dict(case=case['name'],protocol=protocol,repeat=repeat,result=outcome,
                                  expected=expected,actual=current,passed=passed,patch=result.patch,
                                  metadata=result.metadata,telemetry=sampler.summarize(begin,end),
                                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    records.append(record)
                    print(json.dumps({k:record[k] for k in ('case','protocol','result','passed','metadata')},ensure_ascii=True),flush=True)
                    store.close()
        summary = dict(dtype=args.dtype,load_seconds=load_seconds,idle=baseline,llm_resident=resident,records=records)
        output = ROOT/'logs'/f'qwen_quality_{args.dtype}_{args.protocol}_{PROMPT_VERSION}.json'
        output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAVED',output,flush=True)
    finally:
        backend.close()
        sampler.stop()
        gc.collect()


if __name__ == '__main__':
    main()
