"""Compare a local ASS reference with timestamp-matched stored ASR/transcript.

Reference text is evaluation-only; never supplied to an ASR/LLM backend.
"""
import argparse
import json
import re
import sqlite3
import unicodedata
from pathlib import Path


def seconds(value):
    h,m,s=value.split(':')
    return int(h)*3600+int(m)*60+float(s)


def read_ass(path,start,end):
    selected=[]
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.startswith('Dialogue:'): continue
        fields=line.split(',',9)
        if len(fields)!=10: raise ValueError('Malformed ASS Dialogue')
        a,b=seconds(fields[1]),seconds(fields[2])
        if a>=start and b<=end:
            text=re.sub(r'\{[^}]*\}','',fields[9]).replace('\\N',' ').replace('\\n',' ').replace('\\h',' ')
            selected.append(text)
        elif a<end and b>start:
            raise ValueError('Reference cue crosses evaluation boundary; choose a complete cue range')
    if not selected: raise ValueError('No reference cues in range')
    return ''.join(selected)


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def character_error_rate(reference,hypothesis):
    a,b=normalized(reference),normalized(hypothesis)
    if not a: raise ValueError('Empty reference')
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(previous[j]+1,row[j-1]+1,previous[j-1]+(x!=y)))
        previous=row
    return dict(reference_characters=len(a),hypothesis_characters=len(b),
                edit_distance=previous[-1],cer=previous[-1]/len(a))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('database',type=Path)
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 0<=args.start<args.end: parser.error('Invalid range')
    reference=read_ass(args.reference,args.start,args.end)
    with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT start_time,end_time,raw_asr_text,current_text FROM segments WHERE start_time>=? AND end_time<=? ORDER BY ordinal',
                        (args.start,args.end)).fetchall()
        failed=[json.loads(r[0]) for r in db.execute("SELECT data FROM events WHERE type='ASR_FAILED'")]
    raw=''.join(r[2] for r in rows);current=''.join(r[3] for r in rows)
    report=dict(range_seconds=[args.start,args.end],segment_ranges=[[r[0],r[1]] for r in rows],
        raw_asr=character_error_rate(reference,raw),corrected=character_error_rate(reference,current),
        asr_failed_in_range=sum(f['start_sample']/f['sample_rate']<args.end and f['end_sample']/f['sample_rate']>args.start for f in failed),
        limitations='Reference is user-edited AI subtitles, not verified ground truth. Punctuation/case/spacing ignored; numbers and product-name formatting still count as differences.',
        reference=reference,raw_text=raw,corrected_text=current)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('reference','raw_text','corrected_text')},ensure_ascii=True))


if __name__=='__main__':main()
