$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) { $pythonExe = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe' }
if (-not (Test-Path $pythonExe)) { $pythonExe = (Get-Command python).Source }
$torchWheel = Join-Path $PSScriptRoot 'wheels\torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
$audioWheel = Join-Path $PSScriptRoot 'wheels\torchaudio-2.5.1+cu121-cp310-cp310-win_amd64.whl'
if ((Test-Path $torchWheel) -and (Test-Path $audioWheel)) {
    # SHA256 values published in the PyTorch official cu121 wheel indexes.
    if ((Get-FileHash -LiteralPath $torchWheel -Algorithm SHA256).Hash -ne '9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f') {
        throw 'PyTorch wheel checksum mismatch or incomplete download.'
    }
    if ((Get-FileHash -LiteralPath $audioWheel -Algorithm SHA256).Hash -ne '68c08cf69f7f39608f4c8d6b87fa054d7c801a19060e89d66003efbceef15641') {
        throw 'Torchaudio wheel checksum mismatch or incomplete download.'
    }
    & $pythonExe -m pip install --force-reinstall --no-deps $torchWheel $audioWheel
} else {
    & $pythonExe -m pip install --force-reinstall --no-deps torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
}
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime installation failed.' }
& $pythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed. Restart the application after resolving the error.' }
