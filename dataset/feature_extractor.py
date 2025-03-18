import torchaudio.transforms as T

def get_feature_extractor(feature_type="mel_spectrogram", sample_rate=16000, n_mels=80, n_mfcc=13):
    """
    Trả về bộ trích xuất đặc trưng theo yêu cầu.

    feature_type: "mfcc", "mel_spectrogram", "spectrogram"
    """
    if feature_type == "mfcc":
        return T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_mels": n_mels})
    elif feature_type == "mel_spectrogram":
        return T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    elif feature_type == "spectrogram":
        return T.Spectrogram(n_fft=400, hop_length=160)
    else:
        raise ValueError(f"Feature type '{feature_type}' không hợp lệ!")
