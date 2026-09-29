"""Local-only regression replay of immutable ASR evidence; never modifies source."""
import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import replace
from evidence import ASREvidence,Hypothesis
from transcript_store import TranscriptStore
from llm_pipeline import MeetingPipeline
from settings import LocalLLMConfig
from qwen_backend import QwenLocalBackend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('database',type=Path)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--dtype',choices=['bf16','int4'],default='bf16')
    parser.add_argument('--window-chars', type=int, default=1024)
    parser.add_argument('--thinking', action='store_true')
    args=parser.parse_args()
    source=sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True)
    rows=source.execute('SELECT e.data FROM segments s JOIN evidence e ON s.asr_evidence_ref=e.segment_id ORDER BY s.ordinal').fetchall()
    source.close()
    if args.limit:rows=rows[:args.limit]
    output=Path('transcripts/replay')/datetime.now().strftime('%Y%m%d_%H%M%S.sqlite3')
    output.parent.mkdir(parents=True,exist_ok=True)
    backend=QwenLocalBackend(replace(LocalLLMConfig.from_environment(),dtype=args.dtype,
        thinking=args.thinking,max_new_tokens=4096 if args.thinking else 1024))
    backend.load()
    store=TranscriptStore(output, window_chars=args.window_chars)
    pipeline=MeetingPipeline(store,backend=backend)
    try:
        for index,(raw,) in enumerate(rows):
            data=json.loads(raw)
            data['nbest']=tuple(Hypothesis(**h) for h in data['nbest'])
            data['timestamps']=tuple(tuple(t) for t in data['timestamps'])
            pipeline.accept(ASREvidence(**data));pipeline.worker.requests.join()
            event=store.last_event('LLM_REQUEST') or {}
            print(json.dumps(dict(segment=index+1,total=len(rows),result=event.get('result'),
                focuses=event.get('focus_count',1),failures=event.get('partial_failures',0),
                detail=event.get('detail'),latency=event.get('latency')),ensure_ascii=True),flush=True)
        pipeline.close()
        store.export_txt(output.with_suffix('.txt'))
        events=[e['data'] for e in store.events() if e['type']=='LLM_REQUEST']
        children=[sub for e in events for sub in e.get('subrequests',[e])]
        attempts=[attempt for child in children for attempt in child.get('attempts',[child])]
        summary=dict(segments=len(rows),punctuation=sum(c in '，。？！：；,.?!:;' for c in store.canonical_text),
            punctuated_segments=sum(any(c in '，。？！：；,.?!:;' for c in s.current_text) for s in store.segments),
            outcomes=dict(Counter(e['result'] for e in events)),
            child_errors=dict(Counter(e.get('detail','unknown') for e in children if e.get('result') not in ('VALID','APPLIED','NO_CHANGE'))),
            targets=len(children),model_calls=len(attempts),window_chars=store.window_chars,
            attempt_errors=dict(Counter(a.get('detail','unknown') for a in attempts if a.get('result') not in ('VALID','APPLIED','NO_CHANGE'))),
            database=str(output),mean_latency=sum(e['latency'] for e in events)/max(1,len(events)))
        output.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    finally:
        pipeline.close();store.close();backend.close()


if __name__=='__main__':main()
