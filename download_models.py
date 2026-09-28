"""Download/verify Paraformer-large from domestic ModelScope; reuse cached files."""
from engines import MODELS, MODEL_ID


def main():
    from modelscope import snapshot_download
    MODELS.mkdir(exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=str(MODELS / 'paraformer'))
    print('Paraformer-large (220M) ready: ' + str(MODELS / 'paraformer'), flush=True)


if __name__ == '__main__':
    main()
