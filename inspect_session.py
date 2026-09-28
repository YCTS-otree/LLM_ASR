"""Read the latest evidence, correction request and applied patch from a local session."""
import argparse
import json
from pathlib import Path
import sqlite3


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('database',type=Path)
    args=parser.parse_args()
    path=args.database.resolve(strict=True)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        queries={
            'last_asr_evidence':'SELECT data FROM evidence ORDER BY rowid DESC LIMIT 1',
            'last_llm_request':"SELECT data FROM events WHERE type='LLM_REQUEST' ORDER BY event_id DESC LIMIT 1",
            'last_applied_patch':"SELECT data FROM revisions WHERE source='LOCAL_LLM' ORDER BY revision_id DESC LIMIT 1",
        }
        result={}
        for key,query in queries.items():
            row=db.execute(query).fetchone()
            result[key]=json.loads(row[0]) if row else None
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
