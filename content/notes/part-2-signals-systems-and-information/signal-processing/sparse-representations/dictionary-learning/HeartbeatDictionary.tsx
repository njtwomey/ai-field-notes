import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { toArray } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dictionaryLearningSteps, sparseCode } from 'aifn-compute/signal/sparse'
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
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { FixedHeight } from '../_shared/FixedHeight'
import { overlapAdd, RECORDED_RATE, recordedTrace, syntheticTrace, useRecording, windows } from '../_shared/heartbeat'

const N = 1500
const STRIDE = 8
/** The learnt atoms are drawn in this many rows, their columns this many samples apart. */
const ATOM_ROWS = 4
const ATOM_GAP = 6
/** The heights of the trace and of each bottom plot. */
const TOP = 360
const BOTTOM = 150
/** The stretch of the trace drawn at the top. */
const SHOWN = 600

/** The dictionary times the codes, L × n. */
function rebuild(D: number[][], X: number[][]): number[][] {
  return D.map((row) => X[0].map((_, c) => row.reduce((acc, d, j) => acc + d * X[j][c], 0)))
}

export function HeartbeatDictionary() {
  const state = useFigureState({
    source: choice(
      [
        { value: 'synthetic', label: 'synthetic beats' },
        { value: 'recorded', label: 'recorded ECG (MIT-BIH record 208)' },
      ],
      'synthetic',
      { label: 'data' },
    ),
    ectopic: float(0.1, {
      ge: 0,
      le: 0.5,
      suggestions: [0, 0.1, 0.3],
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
      le: 280,
      suggestions: [0, 60, 170, 270],
      label: 'start of the stretch (s)',
      when: (v) => v.source === 'recorded',
    }),
    L: choice([24, 32, 48, 64], 32, { label: 'window length L' }),
    k: int(24, { ge: 2, le: 40, suggestions: [8, 16, 24, 32], label: 'atoms k' }),
    s: int(3, { ge: 1, le: 6, label: 'atoms per window s' }),
    update: choice(
      [
        { value: 'ksvd', label: 'K-SVD' },
        { value: 'mod', label: 'method of optimal directions' },
      ],
      'ksvd',
      { label: 'dictionary update' },
    ),
    rounds: int(15, { ge: 1, le: 40, suggestions: [5, 15, 30], label: 'rounds' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    w: slider(0, SHOWN - 96, 120, { step: STRIDE, onChart: true }),
  })
  const { source, ectopic, noise, wander, start, L, k, s, update, rounds, seed } = state
  const recorded = source === 'recorded'
  const recording = useRecording(recorded)

  const data = useMemo(() => {
    if (recorded) return recording ? recordedTrace(recording, N, start * RECORDED_RATE) : null
    return syntheticTrace(N, seed, { ectopic, noise, wander })
  }, [recorded, recording, start, seed, ectopic, noise, wander])

  const run = useComputed(() => {
    if (!data) return null
    const win = windows(data.x, L, STRIDE)
    const alg = dictionaryLearningSteps(win.Y, { atoms: k, sparsity: Math.min(s, k), update })
    const tr = trace(alg, undefined, rounds, { stream: stream(`dictionary-learning/heartbeat/train/${seed}`) })
    const states = tr.steps.map((st) => ({ D: toArray(st.D) as number[][], objective: st.objective }))
    // Atoms in order of use at the end, most used first; both updates keep each atom's index across rounds.
    const X = toArray(tr.final.X) as number[][]
    const use = X.map((row) => row.filter((v) => v !== 0).length)
    const order = Array.from({ length: k }, (_, j) => j).sort((a, b) => use[b] - use[a] || a - b)
    return { win, states, order, use }
  }, [data, L, k, s, update, rounds, seed])

  const key = `${source}|${start}|${ectopic}|${noise}|${wander}|${L}|${k}|${s}|${update}|${rounds}|${seed}`
  const [pos, setPos] = useState({ key, step: 0 })
  const value = run.value
  const step = value && pos.key === key ? Math.min(pos.step, value.states.length - 1) : 0

  const coded = useMemo(() => {
    // `states` is indexed, so it has to be checked rather than assumed. An
    // empty run makes `step` -1 through the clamp above, and reading `.D` off
    // the undefined that comes back takes the whole page down rather than the
    // one figure.
    if (!value || !data || !value.states[step]) return null
    const D = value.states[step].D
    const X = toArray(sparseCode(D, value.win.Y, { method: 'omp', sparsity: Math.min(s, k) }).X) as number[][]
    const rebuilt = overlapAdd(N, L, value.win.starts, value.win.means, rebuild(D, X))
    const rms = (ref: readonly number[]) =>
      Math.sqrt(rebuilt.reduce((acc, v, t) => acc + (Number.isFinite(v) ? (v - ref[t]) ** 2 : 0), 0) / N)
    return { D, X, trace: rebuilt, errNoisy: rms(data.x), errClean: data.clean ? rms(data.clean) : NaN }
  }, [value, data, step, s, k, L])

  // The window under the handle, snapped to a training window.
  const c = Math.min(Math.round(state.w / STRIDE), (value?.win.starts.length ?? 1) - 1)
  const window = useMemo(() => {
    if (!value || !coded) return null
    const t0 = value.win.starts[c]
    const mean = value.win.means[c]
    const idx = Array.from({ length: L }, (_, i) => t0 + i)
    const used = coded.X.map((row, j) => ({ j, coef: row[c] })).filter((a) => a.coef !== 0)
    const parts = used.map((a) => ({ ...a, y: coded.D.map((row) => row[a.j] * a.coef + mean) }))
    const fit = idx.map((_, i) => mean + used.reduce((acc, a) => acc + coded.D[i][a.j] * a.coef, 0))
    return { t0, idx, y: idx.map((t) => data!.x[t]), fit, parts, used: new Set(used.map((a) => a.j)) }
  }, [value, coded, c, L, data])

  // The atoms in a grid of ATOM_ROWS rows, each scaled to fit its cell, most used first down the columns.
  const atomRows = useMemo(() => {
    if (!value || !coded) return []
    return value.order.map((j, r) => {
      const col = coded.D.map((row) => row[j])
      const peak = Math.max(...col.map(Math.abs), 1e-12)
      const across = Math.floor(r / ATOM_ROWS) * (L + ATOM_GAP)
      return {
        j,
        x: col.map((_, i) => across + i),
        y: col.map((v) => -(r % ATOM_ROWS) + (0.42 * v) / peak),
        label: `atom ${r + 1}, used by ${value.use[j]} windows`,
      }
    })
  }, [value, coded, L])
  const atomColumns = Math.ceil(k / ATOM_ROWS)

  const shownT = useMemo(() => Array.from({ length: SHOWN }, (_, t) => t), [])
  const tx = useAxis({ label: 'sample t', range: [0, SHOWN - 1] })
  const ty = useAxis({ label: 'amplitude', key: key })
  const ax = useAxis({
    label: 'learnt atoms, most used first (down each column)',
    range: [-1, atomColumns * (L + ATOM_GAP) - ATOM_GAP],
    format: () => '',
    key: `${k}|${L}`,
  })
  const ay = useAxis({ format: () => '', range: [-ATOM_ROWS + 0.5, 0.5] })
  const wx = useAxis({ label: 'sample t', range: [c * STRIDE, c * STRIDE + L - 1], key: `${c}|${L}` })
  const wy = useAxis({ label: 'amplitude', key: `${c}|${key}` })
  const ox = useAxis({ label: 'round', integer: true, range: [0, undefined] })
  const oy = useAxis({ label: '½‖Y − DX‖²', log: true })

  const states = value?.states ?? []
  const current = states[step]
  return (
    <Figure
      title="A dictionary learnt from a heartbeat"
      state={state}
      defaultSize="XL"
      caption={`Top: ${SHOWN} samples of an electrocardiogram (grey) and its rebuild (blue): every window of ${L} samples, taken every ${STRIDE} samples less its mean, is coded with s atoms by orthogonal matching pursuit, and the overlapping rebuilds are averaged. The synthetic trace has beats of 54 to 66 samples, a fraction of them ectopic (wide, with no P wave), plus baseline wander and noise. The recorded trace is lead MLII of MIT-BIH Arrhythmia Database record 208 (in mV), decimated from 360 Hz to 90 Hz so that a beat spans about 53 samples; about one beat in five is a premature ventricular beat. Drag the dashed window to choose one. Middle: the learnt atoms, each scaled to fit its cell, in order of use down each column; the atoms the chosen window uses are coloured. Bottom: the chosen window (grey dots), its rebuild (dashed) and the s atoms that make it, each times its coefficient (orange). Right: the training objective by round. Play the rounds: the initial atoms are random windows, and the learnt ones become the QRS spike, the T wave, the P wave and the ectopic shape. Many atoms are the same spike at different shifts, because a window can start anywhere in a beat; convolutional sparse coding shares one atom across all shifts instead. Once the atoms are learnt, the rebuild keeps the beats and removes part of the noise: its RMS error against the clean trace falls below the noise level σ, because white noise is not sparse in these atoms.`}
      controls={
        <Player
          value={step}
          onChange={(t) => setPos({ key, step: t })}
          count={Math.max(states.length, 1)}
          label="round"
          format={(t) => (t === 0 ? 'initial: random windows' : `round ${t} of ${states.length - 1}`)}
        />
      }
      readouts={
        <>
          <Readout label="training objective" value={current ? formatNumber(current.objective) : '–'} />
          <Readout label="RMS difference from the noisy trace" value={coded ? formatNumber(coded.errNoisy) : '–'} />
          {!recorded && (
            <Readout label="RMS error against the clean trace" value={coded ? formatNumber(coded.errClean) : '–'} />
          )}
          <Readout label="training windows" value={value ? String(value.win.starts.length) : '–'} />
        </>
      }
    >
      {!data || !value || !coded || !window ? (
        <p className="py-12 text-center text-sm text-muted-foreground">Loading the recording…</p>
      ) : (
        // A: the trace across the top; B: the atoms and D: the chosen window, two columns wide; C: the objective, two
        // rows tall on the right.
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="md:col-span-3">
            <FixedHeight height={TOP}>
              <Plot x={tx} y={ty}>
                <Curve name="trace" x={shownT} y={data.x.slice(0, SHOWN)} muted width={1} />
                <Curve name="rebuild" x={shownT} y={coded.trace.slice(0, SHOWN)} slot={0} width={1.5} />
                <Annotation x={window.t0} dashed text="window" />
                <Annotation x={window.t0 + L - 1} dashed />
                <Handle {...state.handle('w', { label: 'window' })} />
              </Plot>
            </FixedHeight>
          </div>
          <div className="md:col-span-2">
            <FixedHeight height={BOTTOM}>
              <Plot x={ax} y={ay} legend={false}>
                {atomRows.map((a) => (
                  <Curve
                    key={a.j}
                    name={a.label}
                    x={a.x}
                    y={a.y}
                    slot={window.used.has(a.j) ? 1 : undefined}
                    muted={!window.used.has(a.j)}
                    width={window.used.has(a.j) ? 2 : 1.25}
                  />
                ))}
              </Plot>
            </FixedHeight>
          </div>
          <div className="md:col-start-3 md:row-span-2 md:row-start-2">
            <FixedHeight height={2 * BOTTOM + 16}>
              <Plot x={ox} y={oy} legend={false}>
                <Curve
                  name="objective"
                  x={states.map((_, t) => t)}
                  y={states.map((st) => st.objective)}
                  slot={0}
                  showPoints
                />
                {current && <Points name="this round" x={[step]} y={[current.objective]} emphasis size={10} live />}
              </Plot>
            </FixedHeight>
          </div>
          <div className="md:col-span-2">
            <FixedHeight height={BOTTOM}>
              <Plot x={wx} y={wy} legend={false}>
                <Points name="window" x={window.idx} y={window.y} muted size={4} />
                {window.parts.map((p) => (
                  <Curve
                    key={p.j}
                    name={`atom ${value.order.indexOf(p.j) + 1} × ${formatNumber(p.coef)}`}
                    x={window.idx}
                    y={p.y}
                    slot={1}
                    width={1}
                  />
                ))}
                <Curve name="rebuild" x={window.idx} y={window.fit} emphasis width={1.5} dashed />
              </Plot>
            </FixedHeight>
          </div>
        </div>
      )}
    </Figure>
  )
}
