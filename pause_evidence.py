"""Conservative, labelled pause observations; never invent character timestamps."""
import math
import unicodedata
import numpy as np

PUNCTUATION_MODES = ('baseline', 'pauses', 'combined')


def energy_pauses(audio, rate, threshold, minimum=.12):
    """20ms RMS windows; low energy is a hint, not a speech/VAD decision."""
    width=max(1,round(rate*.02))
    spans=[];start=None
    for offset in range(0,len(audio),width):
        block=audio[offset:offset+width]
        quiet=bool(np.isfinite(block).all() and float(np.sqrt(np.mean(block*block)))<threshold)
        if quiet and start is None:start=offset
        if not quiet and start is not None:
            if (offset-start)/rate>=minimum:spans.append((start/rate,offset/rate))
            start=None
    if start is not None and (len(audio)-start)/rate>=minimum:spans.append((start/rate,len(audio)/rate))
    return tuple(spans)


def pause_hints(evidence):
    """Anchors refer to immutable ASR words, NOT current transcript offsets."""
    best=evidence['best_text']
    raw=evidence.get('raw_result')
    top=raw[0] if isinstance(raw,list) and raw else {}
    tokens=top.get('text','').split() if isinstance(top,dict) else []
    times=evidence.get('timestamps') or []
    plain=lambda s: ''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith('P')).casefold()
    aligned=bool(tokens and len(tokens)==len(times) and plain(''.join(tokens))==plain(best))
    boundaries=[]
    if aligned:
        for i in range(1,len(tokens)):
            boundaries.append(((times[i-1][1]+times[i][0])/2,i))
    hints=[]
    spans=evidence.get('low_energy_spans') or []
    # The full measurements remain in evidence; keep prompt size bounded.
    selected=sorted(sorted(spans,key=lambda p:p[1]-p[0],reverse=True)[:12])
    for start,end in selected:
        if not (math.isfinite(start) and math.isfinite(end) and 0<=start<end):continue
        hint=dict(kind='low_energy_estimate',start_s=round(evidence['start_time']+start,3),
                  end_s=round(evidence['start_time']+end,3),duration_ms=round((end-start)*1000),
                  alignment='time_only')
        if boundaries:
            at,index=min(boundaries,key=lambda pair:abs(pair[0]-(start+end)/2))
            if abs(at-(start+end)/2)<=.35:
                hint.update(alignment='approximate_raw_asr_anchor',
                    before=''.join(tokens[:index])[-16:],after=''.join(tokens[index:])[:16])
        hints.append(hint)
    if aligned and not spans:
        for i in range(1,len(times)):
            start,end=times[i-1][1],times[i][0]
            if end-start>=.12:
                hints.append(dict(kind='asr_timestamp_gap',duration_ms=round((end-start)*1000),
                    start_s=round(evidence['start_time']+start,3),end_s=round(evidence['start_time']+end,3),
                    alignment='approximate_raw_asr_anchor',before=''.join(tokens[:i])[-16:],after=''.join(tokens[i:])[:16]))
        hints=sorted(sorted(hints,key=lambda h:h['duration_ms'],reverse=True)[:12],key=lambda h:h['start_s'])
    return dict(segment_id=evidence['segment_id'],raw_asr=best,
        start_s=evidence['start_time'],end_s=evidence['end_time'],
        end_reason=evidence['end_reason'],trailing_low_energy_ms=evidence.get('pause_after_ms'),
        rms_threshold=evidence.get('pause_threshold'),word_alignment_available=aligned,
        internal_pauses=hints)
