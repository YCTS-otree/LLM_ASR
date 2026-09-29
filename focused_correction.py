"""Bound model output by assigning short, exact edit spans; no Store mutation."""
from dataclasses import replace
import re
from transcript_store import TranscriptSnapshot

FOCUS_CHARS = 128


def focused_contexts(context):
    s=context.snapshot
    # Revisit ALL editable history after new evidence arrives. Read spans overlap;
    # write spans have a single owner, so context is never appended twice.
    start=s.window_start
    targets=[]
    while start<s.window_end:
        end=min(start+FOCUS_CHARS,s.window_end)
        if end<s.window_end:
            boundary=max((s.text.rfind(mark,start+FOCUS_CHARS//2,end) for mark in '。？！；，'),default=-1)
            if boundary>=0:end=boundary+1
        # Avoid cutting a Latin word/number in half when a nearby boundary exists.
        if end<s.window_end and s.text[end-1].isascii() and s.text[end-1].isalnum() and s.text[end].isascii() and s.text[end].isalnum():
            match=re.search(r'[A-Za-z0-9]+$',s.text[start:end])
            if match and match.start()>0:end=start+match.start()
        right=s.text[end:s.window_end]
        evidence=replace(context.evidence,end_reason='context_slice') if right else context.evidence
        sources=tuple(item for item in context.segments if
                      item.get('end',s.window_end)>start-s.window_start and item.get('start',0)<end-s.window_start)
        targets.append(replace(context,snapshot=TranscriptSnapshot(s.revision,s.text[:end],start,end),
            readable_context=s.text[max(0,s.window_start-256):start],readable_right=right,evidence=evidence,segments=sources,focused=True))
        start=end
    return targets
