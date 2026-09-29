"""Independent local punctuation; never changes the recognized words."""
import unicodedata


def lexical(text):
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


class BaselinePunctuation:
    def __init__(self, path):
        from funasr import AutoModel
        if not (path / 'model.pt').is_file():
            raise FileNotFoundError('缺少基础标点模型，请运行 download_models.py --punctuation-only（国内源）')
        self.model = AutoModel(model=str(path), device='cpu', disable_update=True,
                               disable_pbar=True, trust_remote_code=False)

    def punctuate(self, text, end_reason):
        if not text.strip():
            return text
        result = self.model.generate(input=text, disable_pbar=True)
        output = result[0]['text']
        old_words,new_words=lexical(text),lexical(output)
        if len(old_words)!=len(new_words) or old_words.casefold()!=new_words.casefold():
            raise ValueError('Punctuation model changed recognized words')
        # CT-Transformer may capitalize an English sentence's first letter.
        # Transfer punctuation/spacing while preserving raw lexical characters.
        original=iter(old_words)
        output=''.join(next(original) if not c.isspace() and not unicodedata.category(c).startswith('P') else c for c in output)
        # Technical boundaries are not sentence boundaries. Internal punctuation
        # remains usable even if the LLM is disabled or fails.
        return output.rstrip('。？！.!?') if end_reason == 'max_duration' else output
