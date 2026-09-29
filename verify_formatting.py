"""Opt-in real local Qwen formatting probe; synthetic text, no credentials/audio."""
import json
from pathlib import Path
from qwen_backend import QwenLocalBackend
from settings import LocalLLMConfig
from evidence import ASREvidence,StabilityBuffer
from transcript_store import TranscriptStore
from llm_pipeline import ContextBuilder
from patches import RevisionEngine


def main():
    backend=QwenLocalBackend(LocalLLMConfig.from_environment())
    backend.load()
    cases=[('芯片采用二纳米工艺。','2nm'),('配备八千毫安时电池。','8000mAh'),
           ('摄像头拥有两亿像素。','2亿像素'),('公司成立于一九九九年。','1999年'),
           ('我们讨论锐龙九千系处理器。','锐龙9000系'),('做事一心一意，遇事三思而后行。','一心一意')]
    results=[]
    try:
        for source,expected in cases:
            store=TranscriptStore()
            try:
                stable=StabilityBuffer().push(ASREvidence('synthetic',0,1,source))
                store.append(stable)
                context=ContextBuilder().build(store,stable)
                result=backend.process(context)
                if result.patch:RevisionEngine(store).apply(result.patch,context.snapshot,'LOCAL_LLM')
                passed=expected in store.canonical_text
                if '三思' in source:passed=passed and '三思而后行' in store.canonical_text
                results.append(dict(source=source,expected=expected,actual=store.canonical_text,passed=passed,metadata=result.metadata))
            finally:store.close()
    finally:backend.close()
    Path('logs').mkdir(exist_ok=True)
    Path('logs/formatting_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=sum(r['passed'] for r in results),total=len(results)),ensure_ascii=True))
    if not all(r['passed'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
