"""Opt-in GPU measurements and timed ASR/LLM/Qt/SQLite soak, using local sample audio.

No microphone or network. Whole-device nvidia-smi power includes background GPU work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import statistics
import time
import threading
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--measure-only', action='store_true', help='Ten-second active phases, no soak or microphone')
    args = parser.parse_args()
    if not 60 <= args.seconds <= 1800:
        parser.error('Use 60..1800 seconds (release validation requires >=600)')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import psutil
    import torch
    import soundfile as sf
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtGui import QFont, QFontDatabase
    from unittest.mock import Mock, patch
    from app import Window
    from engines import Engine, ROOT, MODELS
    from settings import LocalLLMConfig
    from qwen_backend import QwenLocalBackend
    from llm_protocol import PROMPT_VERSION
    from llm_pipeline import MeetingPipeline, CorrectionContext
    from transcript_store import TranscriptStore
    from evidence import AudioSegment, ASREvidence, Hypothesis, StabilityBuffer
    from audio_pipeline import to_16k
    from gpu_telemetry import GPUSampler
    from qwen_cases import CASES
    from observability import configure_logging
    configure_logging(ROOT/'logs')
    sampler = GPUSampler().start()
    proc = psutil.Process()
    def memory():
        torch.cuda.synchronize()
        return dict(allocated_mib=torch.cuda.memory_allocated()/2**20,
                    reserved_mib=torch.cuda.memory_reserved()/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    rss_mib=proc.memory_info().rss/2**20)
    def measured(fn):
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        result=fn()
        torch.cuda.synchronize()
        end=time.perf_counter()
        return result,dict(seconds=end-start,**memory(),**sampler.summarize(start,end))
    def sustained(fn):
        deadline=time.perf_counter()+10
        results=[]
        while time.perf_counter()<deadline:
            results.append(fn())
        return results
    report=dict(prompt_version=PROMPT_VERSION,seconds_requested=args.seconds)
    _,report['idle']=measured(lambda:time.sleep(2))
    audio,rate=sf.read(MODELS/'paraformer'/'example'/'asr_example.wav',dtype='float32')
    if audio.ndim==2: audio=audio.mean(axis=1)
    audio=to_16k(audio,rate)
    asr=Engine('cuda','fp16')
    _,report['asr_resident']=measured(lambda:time.sleep(2))
    _,report['asr_inference']=measured(lambda:sustained(lambda:asr.transcribe(AudioSegment.from_audio(audio))))
    asr.close()
    del asr
    gc.collect()
    torch.cuda.empty_cache()
    backend=QwenLocalBackend(LocalLLMConfig.from_environment())
    backend.load()
    _,report['llm_resident']=measured(lambda:time.sleep(2))
    case=CASES[0]
    probe=TranscriptStore()
    evidence=ASREvidence('probe',0,1,case['text'],tuple(Hypothesis(i+1,t,-i-1.) for i,t in enumerate(case['candidates'])))
    probe.append(StabilityBuffer().push(evidence))
    context=CorrectionContext(probe.snapshot(),replace(evidence,best_text=case['best']))
    llm_results,report['llm_inference']=measured(lambda:sustained(lambda:backend.process(context)))
    report['llm_requests']=[dict(r.metadata,error=r.error) for r in llm_results]
    asr=Engine('cuda','fp16')
    _,report['combined_resident']=measured(lambda:time.sleep(2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        def concurrent():
            futures=[pool.submit(asr.transcribe,AudioSegment.from_audio(audio)),pool.submit(backend.process,context)]
            return [f.result() for f in futures]
        _,report['concurrent_inference']=measured(lambda:sustained(concurrent))
    probe.close()
    if args.measure_only:
        output=ROOT/'logs'/'local_gpu_benchmark.json'
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        asr.close()
        backend.close()
        sampler.stop()
        print('SAVED',output,flush=True)
        return
    qt=QApplication([])
    font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    if font>=0: qt.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0],10))
    with patch('app.input_devices',return_value=([(1,'Replay (no microphone)','Test')],1)):
        window=Window()
    class Bridge(QObject):
        changed=Signal()
    bridge=Bridge()
    path=ROOT/'transcripts'/'verification'/f'soak_{uuid.uuid4().hex}.sqlite3'
    path.parent.mkdir(parents=True,exist_ok=True)
    store=TranscriptStore(path)
    window.session=Mock(store=store)
    window.session.isRunning.return_value=False
    bridge.changed.connect(window.refresh_transcript)
    pipeline=MeetingPipeline(store,bridge.changed.emit,backend)
    window.llm_state.setText('Qwen3.5-2B · Ready · 本地音频回放稳定性测试')
    window.show()
    stop=threading.Event()
    errors=[]
    latencies=[]
    def producer():
        next_run=time.monotonic()
        ordinal=0
        try:
            while not stop.is_set():
                start=time.perf_counter()
                begin_sample=ordinal*10*16000
                segment=AudioSegment(uuid.uuid4().hex,begin_sample,begin_sample+len(audio),
                                     16000,'silence',audio)
                item=asr.transcribe(segment)
                latencies.append(time.perf_counter()-start)
                pipeline.accept(item)
                ordinal+=1
                next_run+=10
                stop.wait(max(0,next_run-time.monotonic()))
        except Exception as exc:
            errors.append(repr(exc))
            print('SOAK_PRODUCER_ERROR',repr(exc),flush=True)
    thread=threading.Thread(target=producer,name='soak-asr')
    thread.start()
    started=time.perf_counter()
    last_tick=started
    next_sample=started
    health=[]
    ui_gaps=[]
    max_queue=0
    print('SOAK_STARTED',flush=True)
    try:
        while time.perf_counter()-started < args.seconds:
            if errors:
                raise RuntimeError(errors)
            qt.processEvents()
            now=time.perf_counter()
            ui_gaps.append(now-last_tick)
            last_tick=now
            max_queue=max(max_queue,pipeline.worker.requests.qsize())
            if now>=next_sample:
                health.append(dict(elapsed=now-started,queue=pipeline.worker.requests.qsize(),
                                   revision=store.revision,llm_load_count=backend.load_count,**memory()))
                next_sample=now+30
                progress=dict(elapsed=now-started,segments=len(store.segments),health=health[-1])
                (ROOT/'logs'/'soak_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print(json.dumps(progress),flush=True)
            time.sleep(.02)
        stop.set()
        thread.join()
        pipeline.close()
        qt.processEvents()
        window.refresh_transcript()
        assert window.output.toPlainText()==store.canonical_text
        with store.lock:
            integrity=store.db.execute('PRAGMA integrity_check').fetchone()[0]
            requests=[json.loads(row[0]) for row in store.db.execute("SELECT data FROM events WHERE type='LLM_REQUEST'")]
        counts={key:sum(r['result']==key for r in requests) for key in {r['result'] for r in requests}}
        report['soak']=dict(elapsed=time.perf_counter()-started,segments=len(store.segments),
            revisions=store.revision,health=health,max_queue=max_queue,request_results=counts,
            stale_ratio=counts.get('STALE',0)/max(1,len(requests)),llm_load_count=backend.load_count,
            asr_reloaded_during_soak=False,sqlite_integrity=integrity,ui_matches_store=True,
            ui_tick_max_seconds=max(ui_gaps),ui_tick_p95_seconds=sorted(ui_gaps)[int(.95*(len(ui_gaps)-1))],
            asr_latency_median=statistics.median(latencies),requests=requests,errors=errors,
            telemetry=sampler.summarize(started,time.perf_counter()),database=str(path))
        assert not errors and integrity=='ok'
        window.grab().save(str(ROOT/'logs'/'soak_ui.png'))
        output=ROOT/'logs'/'local_benchmark.json'
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAVED',output,flush=True)
    finally:
        stop.set()
        thread.join()
        pipeline.close()
        window.session=None
        window.close()
        store.close()
        asr.close()
        backend.close()
        sampler.stop()


if __name__=='__main__':
    main()
