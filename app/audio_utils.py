import io
import os
import random
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_audio(audio_source: Union[str, Path, bytes, io.BytesIO], target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Tải âm thanh đa định dạng (WAV, MP3, FLAC, OGG, ...) từ đường dẫn file hoặc từ byte stream in-memory.
    Chuyển đổi đa kênh sang đơn kênh (mono) và resample về target_sr (mặc định 16000 Hz).
    """
    if isinstance(audio_source, (str, Path)):
        if not os.path.exists(audio_source):
            raise FileNotFoundError(f"Audio file '{audio_source}' not found.")
        try:
            data, sr = sf.read(str(audio_source))
            wav = torch.from_numpy(data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.t()
        except Exception:
            wav, sr = torchaudio.load(str(audio_source))
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
    elif isinstance(audio_source, (bytes, io.BytesIO)):
        buf = io.BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source
        buf.seek(0)
        try:
            data, sr = sf.read(buf)
            wav = torch.from_numpy(data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.t()
        except Exception:
            buf.seek(0)
            wav, sr = torchaudio.load(buf)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

    # Convert to mono if multi-channel
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if sample rate doesn't match
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    return wav, sr


def extract_fbank(
    waveform: torch.Tensor,
    sr: int = 16000,
    num_mel_bins: int = 128,
    target_length: int = 512,
    is_train: bool = False
) -> torch.Tensor:
    """
    Trích xuất log-mel filterbank (theo chuẩn AST / Kaldi / train.py).
    Kích thước đầu ra: (1, num_mel_bins, target_length), ví dụ (1, 128, 512).
    """
    # Remove DC bias
    waveform = waveform - waveform.mean()

    # Compute Kaldi log-mel filterbank
    # Frame shift 10ms, frame length 25ms, Hanning window
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10
    )  # shape: (time_frames, num_mel_bins)

    n_frames = fbank.shape[0]

    # Pad or cut along the time dimension
    if n_frames < target_length:
        p = target_length - n_frames
        pad = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = pad(fbank)
    elif n_frames > target_length:
        if is_train:
            start = random.randint(0, n_frames - target_length)
            fbank = fbank[start:start + target_length, :]
        else:
            # Center crop during inference / validation
            start = (n_frames - target_length) // 2
            fbank = fbank[start:start + target_length, :]

    # AST style per-sample normalization: (x - mean) / (2 * std + eps)
    fbank = (fbank - fbank.mean()) / (2 * (fbank.std() + 1e-6))

    # Transpose to (num_mel_bins, target_length) and add channel dim -> (1, num_mel_bins, target_length)
    spec = fbank.transpose(0, 1).unsqueeze(0)
    return spec


def audio_to_fbank_tensor(
    audio_source: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = 16000,
    num_mel_bins: int = 128,
    target_length: int = 512
) -> torch.Tensor:
    """
    Pipeline hoàn chỉnh chuyển đổi file âm thanh thô thành 4D Tensor (1, 1, 128, 512)
    sẵn sàng đưa vào mô hình UniRepLKNet-DINOv3.
    """
    wav, sr = load_audio(audio_source, target_sr=target_sr)
    spec = extract_fbank(
        wav,
        sr=sr,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        is_train=False
    )
    # Output shape: (1, 1, num_mel_bins, target_length)
    return spec.unsqueeze(0)
