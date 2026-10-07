import { useEffect, useState } from 'react'
import { convolve } from 'aifn-compute/foundation/convolution'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { syntheticEcg, type SyntheticEcgOptions } from 'aifn-methods/data/signals'

/** A beat in a trace: where it is (its start for synthetic beats, its R peak for recorded ones) and whether it is ectopic. */
export type Beat = { start: number; ectopic: boolean }

/** A stretch of electrocardiogram for the heartbeat figures; `clean` is known only for synthetic traces. */
export type HeartbeatTrace = { x: number[]; clean: number[] | null; beats: Beat[] }

/**
 * A synthetic trace of n samples from `syntheticEcg` (beats of 54 to 66 samples at 60 Hz, a fraction ectopic, plus
 * baseline wander and noise). The stream key keeps the traces these figures have always drawn.
 */
export function syntheticTrace(
  n: number,
  seed: number,
  options: Pick<SyntheticEcgOptions, 'ectopic' | 'noise' | 'wander'>,
): HeartbeatTrace {
  const ecg = syntheticEcg(stream(`dictionary-learning/heartbeat/${seed}`), { n, ...options })
  return {
    x: Array.from(toFlat(ecg.y)),
    clean: Array.from(toFlat(ecg.clean)),
    beats: ecg.beats.filter((b) => b.start < n),
  }
}

/** The recording is decimated by this factor, from 360 Hz to 90 Hz: about 53 samples per beat, like the synthetic trace. */
export const RECORDED_DECIMATE = 4
/** The recording's sample rate after decimation, in Hz. */
export const RECORDED_RATE = 360 / RECORDED_DECIMATE

/** The whole recording: five minutes of lead MLII (mV) of MIT-BIH record 208, and its annotated beats. */
export type Recording = { x: number[]; beats: Beat[] }

let recording: Promise<Recording> | null = null

/** Loads the recording the first time it is asked for: its module holds the whole record (294 KB). */
function loadRecording(): Promise<Recording> {
  recording ??= import('aifn-methods/data/real/ecg')
    .then(({ mitBihEcg }) => {
      const ecg = mitBihEcg({ decimate: RECORDED_DECIMATE })
      return {
        x: Array.from(toFlat(ecg.signal.data)),
        // Premature ventricular beats ('V') are the ectopic beats; the few fusion and unclassified beats are not.
        beats: ecg.beats.map((b) => ({ start: b.sample, ectopic: b.symbol === 'V' })),
      }
    })
    // Forget a failed load, so the next attempt retries. Caching the rejected
    // promise instead means one bad fetch, or one stale build of the data
    // package, leaves the recorded source dead for the life of the tab.
    .catch((error) => {
      recording = null
      throw error
    })
  return recording
}

/** The recording once it has loaded, or null; it is loaded only when `enabled`. */
export function useRecording(enabled: boolean): Recording | null {
  const [value, setValue] = useState<Recording | null>(null)
  useEffect(() => {
    if (!enabled || value) return
    let live = true
    // Reported rather than swallowed, and not rethrown: the figure has a
    // synthetic source that works without this, so a failed load should leave
    // the reader on that rather than taking the page down.
    loadRecording().then(
      (r) => live && setValue(r),
      (error) => console.error('heartbeat: the recording failed to load', error),
    )
    return () => {
      live = false
    }
  }, [enabled, value])
  return enabled ? value : null
}

/** n samples of the recording from sample `start`, with the beats inside them. */
export function recordedTrace(rec: Recording, n: number, start: number): HeartbeatTrace {
  const from = Math.max(0, Math.min(Math.round(start), rec.x.length - n))
  return {
    x: rec.x.slice(from, from + n),
    clean: null,
    beats: rec.beats
      .filter((b) => b.start >= from && b.start < from + n)
      .map((b) => ({ start: b.start - from, ectopic: b.ectopic })),
  }
}

/**
 * Windows of length L every `stride` samples, each less its mean, as the columns of an L × n matrix (row-major,
 * number[][] with L rows). The mean of each window is returned so a rebuild can add it back.
 */
export function windows(x: readonly number[], L: number, stride: number) {
  const starts: number[] = []
  for (let t = 0; t + L <= x.length; t += stride) starts.push(t)
  const means = starts.map((t) => x.slice(t, t + L).reduce((a, b) => a + b, 0) / L)
  const Y = Array.from({ length: L }, (_, i) => starts.map((t, c) => x[t + i] - means[c]))
  return { Y, starts, means }
}

/**
 * The overlap-add rebuild of a trace from coded windows: every sample is the average of the rebuilt windows that
 * cover it, each window's mean added back. Samples no window covers are NaN.
 */
export function overlapAdd(n: number, L: number, starts: readonly number[], means: readonly number[], R: number[][]) {
  const sum = new Array<number>(n).fill(0)
  const count = new Array<number>(n).fill(0)
  starts.forEach((t, c) => {
    for (let i = 0; i < L; i++) {
      sum[t + i] += R[i][c] + means[c]
      count[t + i] += 1
    }
  })
  return sum.map((v, t) => (count[t] ? v / count[t] : NaN))
}

/**
 * The trace less a centred moving average of `width` samples (odd), the window shrunk at the ends: a high-pass filter
 * that removes slow baseline wander and keeps the waves of each beat.
 */
export function removeWander(x: readonly number[], width: number): number[] {
  const half = (width - 1) / 2
  const sums = toFlat(convolve(Array.from(x), new Array<number>(width).fill(1), { mode: 'same' }) as Tensor)
  return x.map((v, n) => v - sums[n] / (Math.min(n + half, x.length - 1) - Math.max(n - half, 0) + 1))
}
