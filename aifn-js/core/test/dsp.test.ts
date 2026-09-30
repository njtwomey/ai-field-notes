import { describe, expect, it } from 'vitest'
import {
  butter,
  cheby1,
  cheby2,
  filtfilt,
  firwin,
  freqz,
  groupDelay,
  kaiserOrder,
  lfilter,
  lfilterZi,
  unwrap,
} from 'aifn/signal/filters'
import { chirp, sampleTimes, tones } from 'aifn-applied/data/signals'
import { convolve, convolve2d, correlate, correlate2d, correlationLags, fftConvolve } from 'aifn/foundation/convolution'
import { cwt, dwt, idwt, wavedec, waveletFilters, wavefun, waverec } from 'aifn/signal/wavelets'
import {
  dct,
  decibels,
  dft,
  fft,
  fft2,
  fftfreq,
  fftshift,
  ifft,
  ifftshift,
  irfft,
  magnitude,
  rfft,
  rfftfreq,
  type ComplexTensor,
} from 'aifn/foundation/fourier'
import { dpss, multitaper, periodogram, spectrogram, stft, welch } from 'aifn/signal/spectral'
import { emd, eemd, siftSteps } from 'aifn/signal/decompositions'
import { envelope, hilbert, instantaneous } from 'aifn/signal/time-frequency'
import { gaussianBlur, sobel } from 'aifn-applied/vision/filters'
import { getWindow, type WindowSpec } from 'aifn/signal/windows'
import { hzToMel, melFilterbank, melToHz, mfcc } from 'aifn-applied/signals/audio'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type Cx = { re: number[] | number[][]; im: number[] | number[][] }
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('dsp')

const flat = (v: Tensor | number[] | number[][]): number[] =>
  'shape' in (v as object) ? toFlat(v as Tensor) : (v as number[]).flat()

function close(actual: Tensor | number[], expected: number[] | number[][], tol = 1e-10) {
  const a = flat(actual)
  const e = (expected as number[]).flat() as number[]
  expect(a.length).toBe(e.length)
  let worst = 0
  const scale = Math.max(1, ...e.map(Math.abs))
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - e[i]))
  expect(worst / scale).toBeLessThan(tol)
}

function closeC(actual: ComplexTensor, expected: Cx, tol = 1e-10) {
  close(actual.re, expected.re, tol)
  close(actual.im, expected.im, tol)
}

const cxIn = (c: Cx): ComplexTensor => ({
  re: fromData(Float64Array.from(c.re as number[])),
  im: fromData(Float64Array.from(c.im as number[])),
})

describe('Fourier transforms against numpy', () => {
  it('fft and ifft of any length (radix-2 and Bluestein), with padding', () => {
    for (const c of F.fft) {
      const x = cxIn(c.x)
      const n = (c.x.re as number[]).length
      closeC(fft(x), c.fft, 1e-11)
      closeC(ifft(x), c.ifft, 1e-11)
      closeC(fft(x, { n: n + 5 }), c.fft_pad, 1e-11)
    }
  })

  it('dft agrees with fft', () => {
    const c = F.fft[4]
    closeC(dft(cxIn(c.x)), c.fft, 1e-10)
  })

  it('rfft, irfft, frequencies and shifts', () => {
    const r = F.rfft
    closeC(rfft(r.x), r.rfft)
    closeC(rfft(r.x, { n: 16 }), r.rfft16)
    close(irfft(rfft(r.x), { n: 9 }), r.irfft)
    close(fftfreq(9, 0.1), r.fftfreq)
    close(rfftfreq(10, 0.1), r.rfftfreq)
    const x = [0, 1, 2, 3, 4]
    expect(toFlat(fftshift(x))).toEqual([3, 4, 0, 1, 2])
    expect(toFlat(ifftshift(fftshift(x)))).toEqual(x)
  })

  it('fft2 matches numpy', () => {
    const rows = F.fft2.x as number[][]
    const t = fromData(Float64Array.from(rows.flat()), [5, 6])
    closeC(fft2(t), F.fft2.fft2)
  })
})

describe('windows against scipy', () => {
  const specs: Record<string, WindowSpec> = {
    hann: 'hann',
    hamming: 'hamming',
    blackman: 'blackman',
    blackmanharris: 'blackmanharris',
    nuttall: 'nuttall',
    flattop: 'flattop',
    bartlett: 'bartlett',
    triangular: 'triangular',
    boxcar: 'boxcar',
    cosine: { name: 'cosine' },
    kaiser: { name: 'kaiser', beta: 8 },
    gaussian: { name: 'gaussian', std: 2.5 },
    tukey: { name: 'tukey', alpha: 0.5 },
  }
  it.each(Object.keys(specs))('%s, symmetric and periodic', (name) => {
    for (const n of [10, 11])
      for (const kind of ['sym', 'periodic']) {
        const w = getWindow(specs[name], n, { periodic: kind === 'periodic' })
        close(w, F.windows[`${name}-${n}-${kind}`], 1e-12)
      }
  })
})

describe('spectral estimation against scipy.signal', () => {
  const S = F.spectral
  it('welch (mean and median, detrending) and periodogram', () => {
    const w = welch(S.x, { fs: 100, nperseg: 128 })
    close(w.f, S.welch.f)
    close(w.psd, S.welch.psd, 1e-10)
    const m = welch(S.x, { fs: 100, nperseg: 100, noverlap: 25, average: 'median', detrend: 'linear' })
    close(m.psd, S.welchMedian.psd, 1e-10)
    const p = periodogram((S.x as number[]).slice(0, 300), { fs: 100, window: 'hann', nfft: 512 })
    close(p.f, S.periodogram.f)
    close(p.psd, S.periodogram.psd, 1e-10)
  })

  it('spectrogram and stft', () => {
    const sg = spectrogram(S.x, { fs: 100, nperseg: 64 })
    close(sg.t, S.spectrogram.t)
    close(sg.power, S.spectrogram.power, 1e-10)
    const st = stft((S.x as number[]).slice(0, 500), { fs: 100, nperseg: 64 })
    close(st.t, S.stft.t)
    close(st.f, S.stft.f)
    closeC(st.Z, S.stft.Z, 1e-10)
  })

  it('dpss tapers and concentrations', () => {
    const d = dpss(64, 3, 5)
    close(d.tapers, F.dpss.tapers, 1e-8)
    close(d.concentrations, F.dpss.ratios, 1e-8)
  })

  it('multitaper integrates to the variance of white noise', () => {
    const x = Array.from({ length: 512 }, (_, i) => Math.sin(i * 1.3) + Math.cos(i * 0.37 + 1))
    const m = multitaper(x, { nw: 3 })
    const df = toFlat(m.f)[1]
    const total = toFlat(m.psd).reduce((a, b) => a + b, 0) * df
    const mean = x.reduce((a, b) => a + b, 0) / x.length
    const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length
    expect(total).toBeCloseTo(variance, 1)
  })
})

describe('filters against scipy.signal', () => {
  it('firwin in every band type, and kaiserord', () => {
    close(firwin(31, 0.3), F.firwin.lowpass, 1e-12)
    close(firwin(31, 0.3, { passZero: false }), F.firwin.highpass, 1e-12)
    close(firwin(41, [0.2, 0.5], { passZero: false, window: { name: 'kaiser', beta: 6 } }), F.firwin.bandpass, 1e-12)
    close(firwin(41, [0.2, 0.5]), F.firwin.bandstop, 1e-12)
    close(firwin(21, 10, { fs: 100, window: 'hann' }), F.firwin.fs, 1e-12)
    const k = kaiserOrder(60, 0.05)
    expect(k.numtaps).toBe(F.firwin.kaiserord[0])
    expect(k.beta).toBeCloseTo(F.firwin.kaiserord[1], 12)
  })

  it('Butterworth and Chebyshev designs', () => {
    const designs: Record<string, () => ReturnType<typeof butter>> = {
      'butter-low': () => butter(4, 0.2),
      'butter-high': () => butter(3, 0.4, { btype: 'highpass' }),
      'butter-band': () => butter(3, [0.2, 0.5], { btype: 'bandpass' }),
      'butter-stop': () => butter(2, [0.2, 0.5], { btype: 'bandstop' }),
      'cheby1-low': () => cheby1(4, 1, 0.3),
      'cheby2-low': () => cheby2(4, 40, 0.3),
      'cheby2-odd': () => cheby2(5, 30, 0.3),
      'butter-fs': () => butter(4, 10, { fs: 100 }),
    }
    for (const [key, design] of Object.entries(designs)) {
      const d = design()
      close(d.b, F.iir[key].b, 1e-9)
      close(d.a, F.iir[key].a, 1e-9)
      expect(d.gain).toBeCloseTo(F.iir[key].k, 10)
    }
  })

  it('lfilter, lfilter_zi and filtfilt', () => {
    const f = F.filtering
    close(lfilter(f.b, f.a, f.x).y, f.lfilter, 1e-10)
    close(lfilterZi(f.b, f.a), f.zi, 1e-10)
    const r = lfilter(f.b, f.a, f.x, { zi: (f.zi as number[]).map((z) => z * f.x[0]) })
    close(r.y, f.lfilterZi.y, 1e-10)
    close(r.zf, f.lfilterZi.zf, 1e-10)
    close(filtfilt(f.b, f.a, f.x), f.filtfilt, 1e-9)
    close(filtfilt(f.b, f.a, f.x, { padtype: 'even', padlen: 20 }), f.filtfiltEven, 1e-9)
    close(lfilter(firwin(15, 0.3), [1], (f.x as number[]).slice(0, 50)).y, f.fir, 1e-12)
  })

  it('freqz, group delay and unwrap', () => {
    const f = F.filtering
    const r = freqz(f.b, f.a, { n: 64 })
    close(r.w, f.freqz.w)
    closeC(r.h, f.freqz.h, 1e-10)
    const g = groupDelay(f.b, f.a, { n: 64 })
    close(g.delay, f.groupDelay.delay, 1e-8)
    close(unwrap(F.unwrap.wrapped), F.unwrap.unwrapped, 1e-12)
  })
})

describe('convolution and correlation', () => {
  const C = F.convolution
  it.each(['full', 'same', 'valid'] as const)('%s mode, direct and FFT, with lags', (mode) => {
    close(convolve(C.u, C.v, { mode }), C[`convolve-${mode}`], 1e-12)
    close(convolve(C.u, C.v, { mode, method: 'fft' }), C[`convolve-${mode}`], 1e-12)
    close(correlate(C.u, C.v, { mode }), C[`correlate-${mode}`], 1e-12)
    expect(toFlat(correlationLags(9, 4, mode))).toEqual(C[`lags-${mode}`])
    expect(toFlat(correlationLags(4, 9, mode))).toEqual(C[`lags-swap-${mode}`])
  })
  it('long inputs take the FFT path', () => {
    close(fftConvolve(C.long.u, C.long.v), C.long.full, 1e-12)
    close(convolve(C.long.u, C.long.v), C.long.full, 1e-12)
  })
})

describe('analytic signal', () => {
  it('hilbert matches scipy for even and odd lengths', () => {
    closeC(hilbert(F.hilbert.even.x), F.hilbert.even.z, 1e-11)
    closeC(hilbert(F.hilbert.odd.x), F.hilbert.odd.z, 1e-11)
  })
  it('an AM tone has its modulation as envelope and its carrier as frequency', () => {
    const t = toFlat(sampleTimes(1024, 1024))
    const x = t.map((s) => (1 + 0.5 * Math.cos(2 * Math.PI * 4 * s)) * Math.cos(2 * Math.PI * 100 * s))
    const env = toFlat(envelope(x))
    const inst = toFlat(instantaneous(x, { fs: 1024 }).frequency)
    for (let i = 200; i < 800; i += 50) {
      expect(env[i]).toBeCloseTo(1 + 0.5 * Math.cos(2 * Math.PI * 4 * t[i]), 2)
      expect(inst[i]).toBeCloseTo(100, 0)
    }
  })
})

describe('wavelets', () => {
  it('filters are orthonormal with the stated vanishing moments', () => {
    for (const name of ['haar', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10'] as const) {
      const f = waveletFilters(name)
      const h = toFlat(f.recLo)
      const g = toFlat(f.recHi)
      expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(Math.SQRT2, 10)
      for (let k = 0; 2 * k < h.length; k++) {
        let s = 0
        for (let n = 2 * k; n < h.length; n++) s += h[n] * h[n - 2 * k]
        expect(s).toBeCloseTo(k === 0 ? 1 : 0, 10)
      }
      for (let p = 0; p < f.vanishingMoments; p++)
        expect(Math.abs(g.reduce((a, gv, n) => a + gv * n ** p, 0))).toBeLessThan(1e-7 * 10 ** p)
    }
  })

  it('dwt preserves energy and idwt / waverec invert exactly', () => {
    const x = Array.from({ length: 64 }, (_, i) => Math.sin(i / 3) + (i % 7) / 5)
    const one = dwt(x, 'db3')
    const energy = (v: number[]) => v.reduce((a, b) => a + b * b, 0)
    expect(energy(toFlat(one.approx)) + energy(toFlat(one.detail))).toBeCloseTo(energy(x), 8)
    close(idwt(one.approx, one.detail, 'db3'), x, 1e-11)
    const d = wavedec(x, 'db4', 3)
    expect(d.details.map((t) => t.shape[0])).toEqual([32, 16, 8])
    close(waverec(d), x, 1e-12)
  })

  it('db2 kills linear trends in the detail coefficients', () => {
    const x = Array.from({ length: 32 }, (_, i) => 3 + 0.5 * i)
    const d = toFlat(dwt(x, 'db2').detail)
    // Periodic wrap-around breaks the trend only in the last coefficient.
    for (let k = 0; k < d.length - 1; k++) expect(Math.abs(d[k])).toBeLessThan(1e-10)
  })

  it('the cascade gives a scaling function with unit integral', () => {
    const w = wavefun('db2', 8)
    const dt = toFlat(w.t)[1]
    expect(toFlat(w.phi).reduce((a, b) => a + b, 0) * dt).toBeCloseTo(1, 6)
  })

  it('the Morlet CWT of a tone peaks at its frequency', () => {
    const fs = 200
    const x = toFlat(tones(sampleTimes(512, fs), [{ frequency: 20 }]))
    const freqs = [5, 10, 20, 40, 60]
    const c = cwt(x, freqs, { fs })
    const mag = toFlat(c.magnitude)
    const mid = freqs.map((_, r) => mag[r * 512 + 256])
    expect(mid.indexOf(Math.max(...mid))).toBe(2)
  })
})

describe('empirical mode decomposition', () => {
  const x = Array.from(
    { length: 512 },
    (_, i) => Math.sin((2 * Math.PI * i) / 16) + 0.8 * Math.sin((2 * Math.PI * i) / 128) + i / 512,
  )

  it('emd reconstructs the signal and separates the fast tone first', () => {
    const d = emd(x, { maxImfs: 3 })
    const [k, n] = d.imfs.shape
    const imfs = toFlat(d.imfs)
    const res = toFlat(d.residue)
    for (let i = 0; i < n; i++) {
      let s = res[i]
      for (let j = 0; j < k; j++) s += imfs[j * n + i]
      expect(s).toBeCloseTo(x[i], 10)
    }
    const first = imfs.slice(100, 400)
    const fast = x.slice(100, 400).map((_, i) => Math.sin((2 * Math.PI * (i + 100)) / 16))
    const err = first.reduce((a, v, i) => a + (v - fast[i]) ** 2, 0) / first.length
    expect(err).toBeLessThan(0.02)
  })

  it('siftSteps follows the trace protocol and ends when the rule holds', () => {
    const opts = { x, rule: { kind: 'fixed', sifts: 6 } as const }
    const t = trace(siftSteps, opts, 20)
    expect(t.meta.stopped).toBe('done')
    expect(t.meta.steps).toBe(6)
    expect(toFlat(seek(siftSteps, opts, 3).h)).toEqual(toFlat(run(siftSteps, opts, 3).h))
    const short = trace(siftSteps, opts, 2, { record: { first: (s) => toFlat(s.h)[0] } })
    const long = trace(siftSteps, opts, 5, { record: { first: (s) => toFlat(s.h)[0] } })
    expect(toFlat(extend(short, siftSteps, opts, 3).series.first)).toEqual(toFlat(long.series.first))
  })

  it('eemd is deterministic in its stream', () => {
    const a = eemd(stream(1), x.slice(0, 128), { trials: 4, maxImfs: 2 })
    const b = eemd(stream(1), x.slice(0, 128), { trials: 4, maxImfs: 2 })
    expect(toFlat(a.imfs)).toEqual(toFlat(b.imfs))
    expect(a.imfs.shape).toEqual([2, 128])
  })
})

describe('audio features', () => {
  it('mel scales invert and the DCT matches scipy', () => {
    for (const f of [0, 300, 1000, 4000]) {
      expect(melToHz(hzToMel(f))).toBeCloseTo(f, 8)
      expect(melToHz(hzToMel(f, 'slaney'), 'slaney')).toBeCloseTo(f, 8)
    }
    expect(hzToMel(1000, 'slaney')).toBeCloseTo(15, 12)
    const m = F.dct.x as number[][]
    close(dct(fromData(Float64Array.from(m.flat()), [3, 8])), F.dct.dct, 1e-12)
  })

  it('filter banks and MFCCs have the right shapes', () => {
    const bank = melFilterbank(20, 512, 16000)
    expect(bank.weights.shape).toEqual([20, 257])
    const x = toFlat(chirp(sampleTimes(4000, 16000), 200, 0.25, 3000))
    const m = mfcc(x, 16000)
    expect(m.mfcc.shape[1]).toBe(13)
    expect(m.mfcc.shape[0]).toBe(m.logMel.shape[0])
    expect(toFlat(m.mfcc).every(Number.isFinite)).toBe(true)
  })
})

describe('image operators against scipy.ndimage', () => {
  const I = F.image
  const img = fromData(Float64Array.from((I.image as number[][]).flat()), [12, 10])
  it.each(['reflect', 'mirror', 'nearest', 'constant', 'wrap'] as const)('correlate2d with %s borders', (border) => {
    close(correlate2d(img, I.kernel, { border }), I[`correlate-${border}`], 1e-12)
  })
  it('convolve2d, Gaussian blur and Sobel', () => {
    close(convolve2d(img, I.kernel), I.convolve, 1e-12)
    close(gaussianBlur(img, 1.5), I.gaussian, 1e-12)
    const s = sobel(img)
    close(s.gx, I.sobelX, 1e-12)
    close(s.gy, I.sobelY, 1e-12)
  })
})

describe('signals and helpers', () => {
  it('chirps match scipy (linear, quadratic, logarithmic)', () => {
    const c = F.chirp
    close(chirp(c.t, 1, 2, 6), c.linear, 1e-12)
    close(chirp(c.t, 1, 2, 6, { method: 'quadratic' }), c.quadratic, 1e-12)
    close(chirp(c.t, 1, 2, 6, { method: 'logarithmic' }), c.logarithmic, 1e-12)
  })
  it('decibels and magnitude', () => {
    expect(toFlat(decibels([1, 10, 100]))).toEqual([0, 10, 20])
    expect(toFlat(decibels([0]))).toEqual([-Infinity])
    expect(toFlat(magnitude({ re: fromData(Float64Array.from([3])), im: fromData(Float64Array.from([4])) }))).toEqual([
      5,
    ])
  })
})
