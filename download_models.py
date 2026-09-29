"""Download/verify Paraformer-large from domestic ModelScope; reuse cached files."""
from engines import MODELS, MODEL_ID


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--punctuation-only',action='store_true')
    args=parser.parse_args()
    from modelscope import snapshot_download
    MODELS.mkdir(exist_ok=True)
    if not args.punctuation_only:
        snapshot_download(MODEL_ID, local_dir=str(MODELS / 'paraformer'))
    snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', local_dir=str(MODELS / 'punctuation'))
    print('Paraformer-large (220M) ready: ' + str(MODELS / 'paraformer'), flush=True)


if __name__ == '__main__':
    main()
