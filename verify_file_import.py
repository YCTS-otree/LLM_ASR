"""Opt-in real file import through Qt; asserts no microphone stream is opened."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--path',type=Path,default=Path('models/paraformer/example/asr_example.wav'))
    parser.add_argument('--dtype',choices=['bf16','int4'],default='bf16')
    parser.add_argument('--mode',choices=['2b','0.8b','both'],default='2b')
    parser.add_argument('--comparison-dtype',choices=['bf16','int4'],default='int4')
    parser.add_argument('--sessions',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--window-chars',type=int,default=1024)
    args=parser.parse_args()
    os.environ['LOCAL_LLM_DTYPE']=args.dtype
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFontDatabase,QFont
    from app import Window
    from observability import configure_logging
    configure_logging(Path('logs'))
    source=args.path.resolve()
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    app=QApplication([])
    font=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    if font>=0:app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0],10))
    window=Window()
    window.writable_window.setValue(args.window_chars)
    window.correction_model.setCurrentIndex(window.correction_model.findData(args.mode))
    window.llm_precision.setCurrentIndex(window.llm_precision.findData(args.dtype))
    window.comparison_precision.setCurrentIndex(window.comparison_precision.findData(args.comparison_dtype))
    window.threshold.setValue(.001)
    window.folder.setText(str(Path('transcripts/import_verification').resolve()))
    errors=[];result={};completed_sessions=[];started=False
    window.failed=errors.append
    def completed():
        nonlocal started
        session=window.session
        try:
            requests=[e['data'] for e in session.store.events() if e['type']=='LLM_REQUEST']
            final=session.store.canonical_text
            result.update(source_unchanged=before==hashlib.sha256(source.read_bytes()).hexdigest(),
                segments=len(session.store.segments),text=final,
                punctuation_count=sum(c in '，。？！：；,.?!:;' for c in final),
                requests=requests,dropped=session.recorder.dropped,
                ui_matches=window.output.toPlainText()==final,database=str(session.store.path),
                mode=next(e['data']['mode'] for e in session.store.events() if e['type']=='SESSION_START'))
            assert result['source_unchanged'] and result['segments']>0 and result['punctuation_count']>0
            assert result['mode']=='file' and result['ui_matches'] and not result['dropped']
            assert Path(session.store.path).with_suffix('.txt').read_text(encoding='utf-8')==final
            result['branches']=[]
            for branch,member,output in zip([session.store,session.comparison_store],window.local_backend.members,[window.output,window.comparison_output]):
                assert member.load_count==1
                events=[e['data'] for e in branch.events() if e['type']=='LLM_REQUEST']
                assert events and all(e['model_id'].endswith(member.config.model_size.upper()) for e in events)
                assert branch.canonical_text==output.toPlainText()==Path(branch.path).with_suffix('.txt').read_text(encoding='utf-8')
                result['branches'].append(dict(model=member.config.model_size,requests=events,text=branch.canonical_text,load_count=member.load_count))
            if session.comparison_store:
                assert [session.store.evidence(s.segment_id) for s in session.store.segments]==[session.comparison_store.evidence(s.segment_id) for s in session.comparison_store.segments]
                a,b=[r['requests'][0] for r in result['branches']]
                result['overlap_seconds']=max(0,min(a['started_monotonic']+a['latency'],b['started_monotonic']+b['latency'])-max(a['started_monotonic'],b['started_monotonic']))
                assert result['overlap_seconds']>0
            import torch
            result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            for width,height in [(900,700),(720,600)]:
                window.resize(width,height);app.processEvents()
                assert window.width()==width and window.height()==height
                window.grab().save(str(Path(f'logs/import_{args.mode}_{width}.png').resolve()))
        except Exception:
            import traceback
            errors.append(traceback.format_exc())
        completed_sessions.append(dict(result))
        if not errors and len(completed_sessions)<args.sessions:
            started=False
            return
        window.close()
    def poll():
        nonlocal started
        if not started and window.local_backend.ready and window.start.isEnabled():
            started=True
            with patch('app.QFileDialog.getOpenFileName',return_value=(str(source),'')):
                window.import_button.click()
            window.session.finished.connect(completed)
    timer=QTimer();timer.timeout.connect(poll);timer.start(50)
    def timeout():
        errors.append('Import timed out')
        window.close()
    QTimer.singleShot(600000,timeout)
    window.show();window.prepare_model()
    with patch('sounddevice.InputStream',side_effect=AssertionError('File import opened microphone')):
        app.exec()
    result.update(passed=not errors,errors=errors,dtype=args.dtype,mode=args.mode,comparison_dtype=args.comparison_dtype)
    result['sessions_completed']=len(completed_sessions)
    Path(f'logs/file_import_{args.mode}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:result.get(key) for key in ('passed','errors','segments','punctuation_count','dropped','ui_matches','source_unchanged','database','peak_allocated_mib')},ensure_ascii=True),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
