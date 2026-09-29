"""Small local reference dictionary; retrieval never performs replacements."""
import json
from pathlib import Path
import unicodedata

DEFAULT_FILE=Path('glossary.json')
EXAMPLE_FILE=Path(__file__).resolve().parent/'glossary.example.json'


def validate_terms(data):
    if type(data) is not list or len(data)>256:raise ValueError('术语库应为最多256项的JSON数组')
    for entry in data:
        if type(entry) is not dict or set(entry)-{'term','aliases','note'}:raise ValueError('每项仅支持term、aliases和note')
        if type(entry.get('term')) is not str or not 1<=len(entry['term'])<=128:raise ValueError('term须为1–128字')
        aliases=entry.get('aliases',[])
        if type(aliases) is not list or len(aliases)>16 or any(type(x) is not str or not 2<=len(x)<=128 for x in aliases):raise ValueError('aliases须为最多16个2–128字的别名')
        if type(entry.get('note','')) is not str or len(entry.get('note',''))>256:raise ValueError('note最多256字')
    return data


def load_terms():
    path=DEFAULT_FILE if DEFAULT_FILE.is_file() else EXAMPLE_FILE
    with path.open('r',encoding='utf-8-sig') as stream:
        raw=stream.read(262145)
    if len(raw)>262144:raise ValueError('术语库超过256KiB字符限制')
    return validate_terms(json.loads(raw))


def relevant_terms(text,terms):
    normalize=lambda s: ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
    text=normalize(text);matches=[]
    for entry in terms:
        needles=[normalize(s) for s in [entry['term']]+entry.get('aliases',[])]
        score=max((len(s) for s in needles if len(s)>=2 and s in text),default=0)
        if score:matches.append((score,entry))
    return [entry for _,entry in sorted(matches,key=lambda x:x[0],reverse=True)[:8]]
