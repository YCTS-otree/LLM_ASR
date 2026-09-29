"""Bounded written-form normalization, not semantic ASR correction."""
import re

DIGITS=dict(zip('零〇一二两三四五六七八九','001223456789'))
UNITS={'十':10,'百':100,'千':1000,'万':10000,'亿':100000000}
NUMBER='零〇一二两三四五六七八九十百千万亿点'
SPEC=re.compile(r'(['+NUMBER+r']+)(纳米(?!比亚)|毫安时|吉赫兹|兆赫兹|亿像素|万像素|像素)')
LABELS={'纳米':'nm','毫安时':'mAh','毫安':'mA','吉赫兹':'GHz','兆赫兹':'MHz','赫兹':'Hz',
        '千瓦':'kW','瓦特':'W','厘米':'cm','毫米':'mm','千克':'kg','公斤':'kg','毫克':'mg','克':'g','伏特':'V','安培':'A',
        '亿像素':'亿像素','万像素':'万像素','像素':'像素'}


def number_value(text):
    if not text or len(text)>20:return None
    if '点' in text:
        if text.count('点')!=1:return None
        left,right=text.split('点');whole=number_value(left)
        if whole is None or not right or any(c not in DIGITS for c in right):return None
        return whole+'.'+''.join(DIGITS[c] for c in right)
    if all(c in DIGITS for c in text):return ''.join(DIGITS[c] for c in text)
    # Reject colloquial abbreviations such as 三千五: could mean3500, not3005.
    if len(text)>1 and text[-1] in DIGITS and text[-2] in '百千万亿':return None
    total=section=number=0;last_small=10000
    for c in text:
        if c in DIGITS:
            if number:return None
            number=int(DIGITS[c])
        elif c in UNITS:
            unit=UNITS[c]
            if unit<10000:
                if unit>=last_small:return None
                section+=(number or 1)*unit;number=0;last_small=unit
            else:
                section+=number
                if unit==10000:section=(section or 1)*unit
                else:total=(total+section or 1)*unit;section=0
                number=0;last_small=10000
        else:return None
    return str(total+section+number)


def normalize_written_numbers(text):
    def specification(match):
        numeral,unit=match.groups()
        # Keep compact pixel magnitudes, rather than expanding to 200000000.
        if unit=='像素' and numeral.endswith(('亿','万')):
            numeral,unit=numeral[:-1],numeral[-1]+'像素'
        value=number_value(numeral)
        return value+LABELS[unit] if value is not None else match.group()
    text=SPEC.sub(specification,text)
    text=re.sub(r'([一二零〇九八七六五四三]{4})年',lambda m: ''.join(DIGITS[c] for c in m[1])+'年',text)
    def model(match):
        value=number_value(match[1])
        return '锐龙'+value+'系' if value is not None else match.group()
    return re.sub(r'锐龙(['+NUMBER+r']+)系',model,text)
