"""Golden values for aifn/dsp, from numpy.fft, scipy.signal, scipy.fft and scipy.ndimage."""

import numpy as np
from scipy import fft as sfft
from scipy import ndimage, signal


def cx(z: np.ndarray) -> dict[str, object]:
    return {"re": np.real(z), "im": np.imag(z)}


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    out: dict[str, object] = {}

    ffts = []
    for n in [1, 2, 7, 8, 13, 100, 128]:
        z = rng.normal(size=n) + 1j * rng.normal(size=n)
        ffts.append(
            {"x": cx(z), "fft": cx(np.fft.fft(z)), "ifft": cx(np.fft.ifft(z)), "fft_pad": cx(np.fft.fft(z, n + 5))}
        )
    out["fft"] = ffts
    real = rng.normal(size=9)
    out["rfft"] = {
        "x": real,
        "rfft": cx(np.fft.rfft(real)),
        "rfft16": cx(np.fft.rfft(real, 16)),
        "irfft": np.fft.irfft(np.fft.rfft(real), 9),
        "fftfreq": np.fft.fftfreq(9, 0.1),
        "rfftfreq": np.fft.rfftfreq(10, 0.1),
    }
    img = rng.normal(size=(5, 6))
    out["fft2"] = {"x": img, "fft2": cx(np.fft.fft2(img))}

    windows = {}
    specs: list[tuple[str, object]] = [
        ("hann", "hann"),
        ("hamming", "hamming"),
        ("blackman", "blackman"),
        ("blackmanharris", "blackmanharris"),
        ("nuttall", "nuttall"),
        ("flattop", "flattop"),
        ("bartlett", "bartlett"),
        ("triangular", "triang"),
        ("boxcar", "boxcar"),
        ("cosine", "cosine"),
        ("kaiser", ("kaiser", 8.0)),
        ("gaussian", ("gaussian", 2.5)),
        ("tukey", ("tukey", 0.5)),
    ]
    for name, spec in specs:
        for n in (10, 11):
            windows[f"{name}-{n}-sym"] = signal.get_window(spec, n, fftbins=False)
            windows[f"{name}-{n}-periodic"] = signal.get_window(spec, n, fftbins=True)
    out["windows"] = windows

    t = np.arange(1000) / 100.0
    x = np.sin(2 * np.pi * 7 * t) + 0.5 * rng.normal(size=t.size)
    f, p = signal.welch(x, fs=100.0, nperseg=128)
    fm, pm = signal.welch(x, fs=100.0, nperseg=100, noverlap=25, average="median", detrend="linear")
    fp, pp = signal.periodogram(x[:300], fs=100.0, window="hann", nfft=512)
    fs_, ts_, sxx = signal.spectrogram(x, fs=100.0, nperseg=64)
    fz, tz, zxx = signal.stft(x[:500], fs=100.0, nperseg=64)
    out["spectral"] = {
        "x": x,
        "welch": {"f": f, "psd": p},
        "welchMedian": {"f": fm, "psd": pm},
        "periodogram": {"f": fp, "psd": pp},
        "spectrogram": {"f": fs_, "t": ts_, "power": sxx},
        "stft": {"f": fz, "t": tz, "Z": cx(zxx)},
    }
    tapers, ratios = signal.windows.dpss(64, 3.0, 5, return_ratios=True)
    out["dpss"] = {"tapers": tapers, "ratios": ratios}

    out["firwin"] = {
        "lowpass": signal.firwin(31, 0.3),
        "highpass": signal.firwin(31, 0.3, pass_zero=False),
        "bandpass": signal.firwin(41, [0.2, 0.5], pass_zero=False, window=("kaiser", 6.0)),
        "bandstop": signal.firwin(41, [0.2, 0.5]),
        "fs": signal.firwin(21, 10.0, fs=100.0, window="hann"),
        "kaiserord": list(signal.kaiserord(60.0, 0.05)),
    }
    iir = {}
    designs = {
        "butter-low": signal.butter(4, 0.2, output="zpk"),
        "butter-high": signal.butter(3, 0.4, btype="highpass", output="zpk"),
        "butter-band": signal.butter(3, [0.2, 0.5], btype="bandpass", output="zpk"),
        "butter-stop": signal.butter(2, [0.2, 0.5], btype="bandstop", output="zpk"),
        "cheby1-low": signal.cheby1(4, 1.0, 0.3, output="zpk"),
        "cheby2-low": signal.cheby2(4, 40.0, 0.3, output="zpk"),
        "cheby2-odd": signal.cheby2(5, 30.0, 0.3, output="zpk"),
        "butter-fs": signal.butter(4, 10.0, fs=100.0, output="zpk"),
    }
    for key, (z, pl, k) in designs.items():
        b, a = signal.zpk2tf(z, pl, k)
        iir[key] = {"b": b, "a": a, "k": k}
    out["iir"] = iir

    b, a = signal.butter(4, 0.2)
    zi = signal.lfilter_zi(b, a)
    y, zf = signal.lfilter(b, a, x[:200], zi=zi * x[0])
    w, h = signal.freqz(b, a, worN=64)
    wg, gd = signal.group_delay((b, a), w=64)
    out["filtering"] = {
        "b": b,
        "a": a,
        "x": x[:200],
        "lfilter": signal.lfilter(b, a, x[:200]),
        "zi": zi,
        "lfilterZi": {"y": y, "zf": zf},
        "filtfilt": signal.filtfilt(b, a, x[:200]),
        "filtfiltEven": signal.filtfilt(b, a, x[:200], padtype="even", padlen=20),
        "fir": signal.lfilter(signal.firwin(15, 0.3), [1.0], x[:50]),
        "freqz": {"w": w, "h": cx(h)},
        "groupDelay": {"w": wg, "delay": gd},
    }
    u = rng.normal(size=9)
    v = rng.normal(size=4)
    conv = {"u": u, "v": v}
    for mode in ("full", "same", "valid"):
        conv[f"convolve-{mode}"] = signal.convolve(u, v, mode=mode)
        conv[f"correlate-{mode}"] = signal.correlate(u, v, mode=mode)
        conv[f"lags-{mode}"] = signal.correlation_lags(9, 4, mode=mode)
        conv[f"lags-swap-{mode}"] = signal.correlation_lags(4, 9, mode=mode)
    long_u = rng.normal(size=300)
    long_v = rng.normal(size=80)
    conv["long"] = {"u": long_u, "v": long_v, "full": signal.convolve(long_u, long_v)}
    out["convolution"] = conv

    out["hilbert"] = {
        "even": {"x": x[:16], "z": cx(signal.hilbert(x[:16]))},
        "odd": {"x": x[:15], "z": cx(signal.hilbert(x[:15]))},
    }
    phase = np.cumsum(rng.uniform(0.5, 2.5, size=30))
    out["unwrap"] = {"wrapped": np.angle(np.exp(1j * phase)), "unwrapped": np.unwrap(np.angle(np.exp(1j * phase)))}
    m = rng.normal(size=(3, 8))
    out["dct"] = {"x": m, "dct": sfft.dct(m, norm="ortho", axis=-1)}

    image = rng.uniform(size=(12, 10))
    kernel = rng.normal(size=(3, 3))
    nd = {"image": image, "kernel": kernel}
    for mode in ("reflect", "mirror", "nearest", "constant", "wrap"):
        nd[f"correlate-{mode}"] = ndimage.correlate(image, kernel, mode=mode)
    nd["convolve"] = ndimage.convolve(image, kernel, mode="reflect")
    nd["gaussian"] = ndimage.gaussian_filter(image, 1.5)
    nd["sobelX"] = ndimage.sobel(image, axis=1)
    nd["sobelY"] = ndimage.sobel(image, axis=0)
    out["image"] = nd

    tt = np.linspace(0, 2, 50)
    out["chirp"] = {
        "t": tt,
        "linear": signal.chirp(tt, 1.0, 2.0, 6.0),
        "quadratic": signal.chirp(tt, 1.0, 2.0, 6.0, method="quadratic", vertex_zero=True),
        "logarithmic": signal.chirp(tt, 1.0, 2.0, 6.0, method="logarithmic"),
    }
    return out
