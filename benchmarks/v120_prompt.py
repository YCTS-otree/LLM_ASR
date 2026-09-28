import json
def build_messages(context, protocol='relative'):
    from pathlib import Path
    system = (Path(__file__).parent / 'v120_system.txt').read_text(encoding='utf-8-sig')
    if protocol == 'relative':
        spec = ('每项为 {"op":"replace|insert|delete","start":整数,"end":整数,"text":"新文本",'
                '"old_text":"被替换的原文"}。start/end 是 writable_window.text 内从0开始的 Python Unicode 字符偏移，'
                'end不包含在内。old_text 必须逐字等于该范围原文；insert 的start=end且old_text为空，delete的text为空。')
    else:
        spec = ('replace/delete 项为 {"op":"replace|delete","old_text":"准确原文","new_text":"新文本"}，'
                'delete的new_text为空。old_text必须在窗口内唯一，可包含少量相邻原文消除歧义。'
                'insert项为 {"op":"insert","anchor":"唯一原文","position":"before|after","text":"插入文本"}。')
    evidence = context.evidence
    data = {
        'base_revision': context.snapshot.revision,
        'protocol': protocol,
        'writable_window': {'text': context.snapshot.writable_text, 'length': len(context.snapshot.writable_text)},
        'readable_context': getattr(context, 'readable_context', ''),
        'asr_evidence': {'segment_id': evidence.segment_id, 'end_reason': evidence.end_reason,
                         'may_continue': evidence.end_reason == 'max_duration', 'best': evidence.best_text,
                         'nbest': [dict(rank=h.rank, text=h.text, score=h.score) for h in evidence.nbest]},
    }
    schema = ('{"base_revision":'+str(context.snapshot.revision)+',"operations":[' +
              ('{"op":"replace","start":整数,"end":整数,"old_text":"原词","text":"正确词"}'
               if protocol == 'relative' else '{"op":"replace","old_text":"原词","new_text":"正确词"}') + ']}')
    # JSON-quoted values keep transcript strings in one data message, including
    # literal newlines/tags. Program-generated instructions follow the data.
    payload = ('只读前文 readable_context: ' + json.dumps(data['readable_context'],ensure_ascii=False) +
               '\n转录 writable_window: ' + json.dumps(data['writable_window']['text'],ensure_ascii=False) +
               '\n新片段 ASR Top1: ' + json.dumps(evidence.best_text,ensure_ascii=False) +
               '\nASR候选: ' + json.dumps(data['asr_evidence']['nbest'],ensure_ascii=False,allow_nan=False) +
               '\n分段信息: ' + json.dumps(dict(end_reason=evidence.end_reason,may_continue=evidence.end_reason=='max_duration')))
    return [{'role': 'system', 'content': system + '\n' + spec},
            {'role': 'user', 'content': payload + '\n请纠正识别错误，输出结构为' + schema +
             '。没有错误则operations为空数组。最多两处最小修改。'}]

