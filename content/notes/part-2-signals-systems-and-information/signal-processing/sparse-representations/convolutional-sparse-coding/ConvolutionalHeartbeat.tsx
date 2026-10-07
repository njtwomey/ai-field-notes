import { useMemo, useState } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { FixedHeight } from '../_shared/FixedHeight'
import { RECORDED_RATE, recordedTrace, removeWander, syntheticTrace, useRecording } from '../_shared/heartbeat'
import { convolutionalDictionaryLearningSteps, convolutionalSynthesis } from 'aifn-compute/signal/sparse'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

/** Samples in the trace: about fifteen beats. */
const N = 900
/** Width of the moving average subtracted to remove baseline wander. */
const DETREND = 121
/** Half-width of the zoom around the chosen time. */
const HALF = 45
/** FISTA steps per round of coding. */
const CODE_STEPS = 25
const FILTER_GAP = 8
const TOP = 300
const BOTTOM = 170

const detrend = (x: readonly number[]) => removeWander(x, DETREND)

export function ConvolutionalHeartbeat() {
  const state = useFigureState({
    source: choice(
      [
        { value: 'synthetic', label: 'synthetic beats' },
        { value: 'recorded', label: 'recorded ECG (MIT-BIH record 208)' },
      ],
      'synthetic',
      { label: 'data' },
    ),
    ectopic: float(0.25, {
      ge: 0,
      le: 0.5,
      suggestions: [0, 0.1, 0.25, 0.4],
      label: 'ectopic beats (fraction)',
      when: (v) => v.source === 'synthetic',
    }),
    noise: float(0.08, {
      ge: 0,
      le: 0.3,
      suggestions: [0, 0.03, 0.08, 0.15],
      label: 'noise σ',
      when: (v) => v.source === 'synthetic',
    }),
    wander: float(0.2, {
      ge: 0,
      le: 1,
      suggestions: [0, 0.2, 0.5],
      label: 'baseline wander',
      when: (v) => v.source === 'synthetic',
    }),
    start: float(170, {
      ge: 0,
      le: 290,
      suggestions: [0, 60, 170, 270],
      label: 'start of the stretch (s)',
      when: (v) => v.source === 'recorded',
    }),
    K: int(3, { ge: 1, le: 5, label: 'filters K' }),
    L: choice([24, 32, 48, 64], 48, { label: 'filter length L' }),
    lambda: float(0.4, { ge: 0.02, le: 2, scale: 'log10', suggestions: [0.1, 0.2, 0.4, 0.8], label: 'penalty λ' }),
    rounds: int(10, { ge: 1, le: 30, suggestions: [5, 10, 20], label: 'rounds' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    t: slider(HALF, N - 1 - HALF, 250, { step: 1, onChart: true }),
  })
  const { source, ectopic, noise, wander, start, K, L, lambda, rounds, seed } = state
  const recorded = source === 'recorded'
  const recording = useRecording(recorded)

  const data = useMemo(() => {
    const ecg = recorded
      ? recording && recordedTrace(recording, N, start * RECORDED_RATE)
      : syntheticTrace(N, seed, { ectopic, noise, wander })
    if (!ecg) return null
    return { y: detrend(ecg.x), clean: ecg.clean && detrend(ecg.clean), beats: ecg.beats }
  }, [recorded, recording, start, seed, ectopic, noise, wander])

  const run = useComputed(
    () => {
      if (!data) return null
      const alg = convolutionalDictionaryLearningSteps(data.y, { filters: K, length: L, lambda, codeSteps: CODE_STEPS })
      const tr = trace(alg, undefined, rounds, { stream: stream(`convolutional-sparse-coding/${seed}`) })
      return tr.steps.map((st) => ({
        d: toArray(st.D) as number[][],
        z: toArray(st.Z) as number[][],
        objective: st.objective,
        nonzeros: st.nonzeros,
      }))
    },
    [data, K, L, lambda, rounds, seed],
    { mode: 'release' },
  )
  const states = run.value ?? []

  const key = `${source}|${start}|${ectopic}|${noise}|${wander}|${K}|${L}|${lambda}|${rounds}|${seed}`
  const [pos, setPos] = useState({ key, step: 0 })
  const step = states.length && pos.key === key ? Math.min(pos.step, states.length - 1) : 0
  const current = states[step]

  const view = useMemo(() => {
    if (!current || !data) return null
    const { d, z } = current
    const rebuilt = toFlat(convolutionalSynthesis(d, z))
    // Each activation is drawn at its filter's peak, so it lines up with the wave it places.
    const peak = d.map((dk) => dk.reduce((best, v, l) => (Math.abs(v) > Math.abs(dk[best]) ? l : best), 0))
    const stems = z.map((zk, k) => {
      const top = Math.max(...zk.map(Math.abs), 1e-12)
      return zk.flatMap((v, m) =>
        v === 0 ? [] : [{ from: [m + peak[k], -k] as const, to: [m + peak[k], -k + (0.8 * v) / top] as const }],
      )
    })
    const filters = d.map((dk, k) => ({ x: dk.map((_, l) => k * (L + FILTER_GAP) + l), y: dk }))
    const rms = (ref: readonly number[]) => Math.sqrt(rebuilt.reduce((a, v, n) => a + (v - ref[n]) ** 2, 0) / N)
    return { rebuilt, stems, filters, rmsNoisy: rms(data.y), rmsClean: data.clean ? rms(data.clean) : NaN }
  }, [current, data, L])

  // The zoom: the copies of each filter placed within the window around t, each scaled by its activation.
  const t = state.t
  const zoom = useMemo(() => {
    if (!current || !data || !view) return null
    const lo = t - HALF
    const hi = t + HALF
    const copies = current.z.flatMap((zk, k) =>
      zk.flatMap((v, m) =>
        v === 0 || m + L - 1 < lo || m > hi
          ? []
          : [{ k, m, x: current.d[k].map((_, l) => m + l), y: current.d[k].map((dv) => v * dv) }],
      ),
    )
    const idx = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
    return { lo, hi, idx, y: idx.map((n) => data.y[n]), fit: idx.map((n) => view.rebuilt[n]), copies }
  }, [current, data, view, t, L])

  const time = useMemo(() => Array.from({ length: N }, (_, n) => n), [])
  const beats = data?.beats?.length ?? N / 60
  const tx = useAxis({ label: 'sample n', range: [0, N - 1] })
  const ty = useAxis({ label: 'amplitude (detrended)', key })
  const ax = useAxis({ label: 'activation maps, one row per filter', range: [0, N - 1] })
  const ay = useAxis({ format: () => '', range: [-K + 0.4, 1], key: K })
  const fx = useAxis({
    label: 'learnt filters',
    format: () => '',
    range: [-2, K * (L + FILTER_GAP) - FILTER_GAP + 1],
    key: `${K}|${L}`,
  })
  const fy = useAxis({ label: 'tap value', key })
  const zx = useAxis({ label: 'sample n', range: [t - HALF, t + HALF], key: t })
  const zy = useAxis({ label: 'amplitude', key: `${t}|${key}` })
  const ox = useAxis({ label: 'round', integer: true, range: [0, undefined] })
  const oy = useAxis({ label: 'objective' })

  return (
    <Figure
      title="Convolutional sparse coding of a heartbeat"
      state={state}
      defaultSize="XL"
      caption={`Top: ${N} samples of an electrocardiogram (synthetic beats, or lead MLII of MIT-BIH Arrhythmia Database record 208 decimated to 90 Hz, in which about one beat in five is a premature ventricular beat) with its baseline wander removed (grey; a moving average over ${DETREND} samples is subtracted) and its rebuild (blue), Σ_k z_k ∗ d_k. Below it: the activation maps z_k, one row per filter in that filter's colour, each stem drawn at its filter's peak so it lines up with the wave it places. Bottom left: the K learnt filters d_k, each of unit norm or less. Bottom middle: a zoom on the dashed window (drag it on the top chart): the trace (grey dots), the copies of each filter placed in the window, each scaled by its activation (coloured), and their sum (dashed). Bottom right: the objective ½‖y − Σ z_k ∗ d_k‖² + λ Σ‖z_k‖₁ by round; each round codes by ${CODE_STEPS} FISTA steps, warm-started, then refits the filters by least squares. Play the rounds: the initial filters are random stretches of the trace; the learnt ones become the normal beat, a QRS complex followed by its T wave (often as two variants that share the normal beats), and the wide ectopic beat, which fires only at the ectopic beats. One filter serves every position, so each beat costs about one activation where a dictionary of windows needs several atoms per window. Lower λ for more, smaller activations that fit finer detail and some of the noise.`}
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={Math.max(states.length, 1)}
          label="round"
          format={(s) => (s === 0 ? 'initial: random stretches' : `round ${s} of ${states.length - 1}`)}
        />
      }
      readouts={
        <>
          <Readout label="objective" value={current ? formatNumber(current.objective) : '–'} />
          <Readout label="non-zeros" value={current ? String(current.nonzeros) : '–'} />
          <Readout label="non-zeros per beat" value={current ? formatNumber(current.nonzeros / beats) : '–'} />
          {!recorded && (
            <Readout label="RMS error against the clean trace" value={view ? formatNumber(view.rmsClean) : '–'} />
          )}
          <Readout label="RMS difference from the trace" value={view ? formatNumber(view.rmsNoisy) : '–'} />
        </>
      }
    >
      {!data || !current || !view || !zoom ? (
        <p className="py-12 text-center text-sm text-muted-foreground">
          {data ? 'Learning…' : 'Loading the recording…'}
        </p>
      ) : (
        <div className="flex flex-col gap-3">
          <FixedHeight height={TOP}>
            <Plot x={tx} y={ty}>
              <Curve name="trace" x={time} y={data.y} muted width={1} />
              <Curve name="rebuild" x={time} y={view.rebuilt} slot={0} width={1.5} />
              <Annotation x={zoom.lo} dashed />
              <Annotation x={zoom.hi} dashed />
              <Handle {...state.handle('t', { label: 'zoom centre' })} />
            </Plot>
          </FixedHeight>
          <FixedHeight height={40 + 36 * K}>
            <Plot x={ax} y={ay} legend={false}>
              {view.stems.map((segments, k) => (
                <Segments key={k} name={`filter ${k + 1}`} segments={segments} slot={k} width={2} />
              ))}
              {current.d.map((_, k) => (
                <Annotation key={`row-${k}`} y={-k} text={`filter ${k + 1}`} muted />
              ))}
              <Annotation x={zoom.lo} dashed />
              <Annotation x={zoom.hi} dashed />
            </Plot>
          </FixedHeight>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <FixedHeight height={BOTTOM}>
              <Plot x={fx} y={fy} legend={false}>
                {view.filters.map((f, k) => (
                  <Curve key={k} name={`filter ${k + 1}`} x={f.x} y={f.y} slot={k} width={2} />
                ))}
              </Plot>
            </FixedHeight>
            <FixedHeight height={BOTTOM}>
              <Plot x={zx} y={zy} legend={false}>
                <Points name="trace" x={zoom.idx} y={zoom.y} muted size={3} />
                {zoom.copies.map((c) => (
                  <Curve
                    key={`${c.k}-${c.m}`}
                    name={`filter ${c.k + 1} at ${c.m}`}
                    x={c.x}
                    y={c.y}
                    slot={c.k}
                    width={1.25}
                  />
                ))}
                <Curve name="rebuild" x={zoom.idx} y={zoom.fit} emphasis width={1.5} dashed />
              </Plot>
            </FixedHeight>
            <FixedHeight height={BOTTOM}>
              <Plot x={ox} y={oy} legend={false}>
                <Curve
                  name="objective"
                  x={states.map((_, s) => s)}
                  y={states.map((s) => s.objective)}
                  slot={0}
                  showPoints
                />
                <Points name="this round" x={[step]} y={[current.objective]} emphasis size={10} live />
              </Plot>
            </FixedHeight>
          </div>
        </div>
      )}
    </Figure>
  )
}
