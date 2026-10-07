import { useMemo } from 'react'
import { dct, idct } from 'aifn-compute/foundation/fourier'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { hardThreshold } from 'aifn-compute/signal/sparse'
import { wavedec, waverec } from 'aifn-compute/signal/wavelets'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const N = 64
const LEVELS = 6
const T = Array.from({ length: N }, (_, i) => i)

const smooth = (i: number) => 1.5 * Math.exp(-(((i - 22) / 7) ** 2)) - 0.9 * Math.exp(-(((i - 46) / 9) ** 2)) + 0.2
const steps = (i: number) => (i < 16 ? 0.8 : i < 40 ? -0.6 : i < 48 ? 1.4 : 0.2)
const SPIKES: Record<number, number> = { 9: 1.6, 23: -1.1, 37: 0.9, 54: -1.4 }

const SIGNALS = {
  smooth: { label: 'smooth (two bumps)', at: smooth },
  steps: { label: 'piecewise constant', at: steps },
  spikes: { label: 'four spikes', at: (i: number) => SPIKES[i] ?? 0 },
  mixed: { label: 'smooth plus spikes', at: (i: number) => smooth(i) + (SPIKES[i] ?? 0) },
} as const
type SignalId = keyof typeof SIGNALS

const BASES = [
  { id: 'spikes', label: 'spikes', slot: 0 },
  { id: 'cosines', label: 'cosines (DCT)', slot: 1 },
  { id: 'haar', label: 'Haar wavelets', slot: 2 },
] as const
type BasisId = (typeof BASES)[number]['id']

/** The coefficients of x in an orthonormal basis, and the synthesis that maps coefficients back to samples. */
function analyse(
  x: number[],
  basis: BasisId,
): { c: number[]; synth: (c: number[]) => number[]; name: (j: number) => string } {
  if (basis === 'spikes') return { c: x, synth: (c) => c, name: (j) => `spike at i = ${j}` }
  if (basis === 'cosines')
    return { c: toFlat(dct(x) as Tensor), synth: (c) => toFlat(idct(tensor(c))), name: (j) => `cosine k = ${j}` }
  const dec = wavedec(x, 'haar', LEVELS)
  const parts = [toFlat(dec.approx), ...dec.details.map((d) => toFlat(d))]
  const synth = (c: number[]) => {
    let at = 0
    const cut = parts.map((p) => {
      const out = c.slice(at, at + p.length)
      at += p.length
      return out
    })
    return toFlat(waverec({ ...dec, approx: tensor(cut[0]), details: cut.slice(1).map((d) => tensor(d)) }).data)
  }
  // Coefficient 0 is the constant; then the details from the finest (width 2) to the coarsest (width 64).
  const name = (j: number) => {
    if (j === 0) return 'constant'
    let at = 1
    for (let level = 1; level < parts.length; level++) {
      const len = parts[level].length
      if (j < at + len) return `Haar, width ${2 ** level}, at i = ${(j - at) * 2 ** level}`
      at += len
    }
    return `Haar ${j}`
  }
  return { c: parts.flat(), synth, name }
}

/** The atoms an approximation uses, largest coefficient first, at most this many drawn. */
const MAX_ATOMS = 8

/** The relative error of the best s-term approximation for s = 0..N: by Parseval, the energy of the dropped tail. */
function errorCurve(c: number[]): number[] {
  const e = c.map((v) => v * v).sort((a, b) => b - a)
  const total = e.reduce((a, b) => a + b, 0)
  const out: number[] = []
  let tail = total
  for (let s = 0; s <= N; s++) {
    out.push(Math.sqrt(Math.max(tail, 0) / total))
    if (s < N) tail -= e[s]
  }
  return out
}

export function BasisSparsity() {
  const state = useFigureState({
    signal: choice(
      (Object.keys(SIGNALS) as SignalId[]).map((id) => ({ value: id, label: SIGNALS[id].label })),
      'smooth',
      { label: 'signal' },
    ),
    basis: choice(
      BASES.map((b) => ({ value: b.id, label: b.label })),
      'cosines',
      { label: 'approximate in' },
    ),
    s: slider(1, N, 10, { step: 1, label: 'terms kept s' }),
  })
  const x = useMemo(() => T.map(SIGNALS[state.signal as SignalId].at), [state.signal])
  const all = useMemo(() => BASES.map((b) => ({ ...b, ...analyse(x, b.id) })), [x])
  const curves = useMemo(() => all.map((b) => errorCurve(b.c)), [all])
  const shown = all.find((b) => b.id === state.basis) ?? all[0]
  const approx = useMemo(() => shown.synth(toFlat(hardThreshold(shown.c, state.s))), [shown, state.s])
  const atoms = useMemo(() => {
    const order = shown.c
      .map((_, j) => j)
      .sort((a, b) => Math.abs(shown.c[b]) - Math.abs(shown.c[a]))
      .slice(0, Math.min(state.s, MAX_ATOMS))
    const curves = order.map((j) => {
      const unit = new Array<number>(N).fill(0)
      unit[j] = shown.c[j]
      return { j, c: shown.c[j], y: shown.synth(unit) }
    })
    // Each atom times its coefficient is scaled to fill its own row, so small coefficients still show their shape.
    return curves.map((a, r) => ({
      ...a,
      y: a.y.map((v) => (0.4 * v) / Math.max(...a.y.map(Math.abs), 1e-12)),
      offset: -r,
      label: `${shown.name(a.j)}, coefficient ${formatNumber(a.c)}`,
    }))
  }, [shown, state.s])
  const sAxis = useMemo(() => Array.from({ length: N + 1 }, (_, s) => s), [])

  const tx = useAxis({ label: 'sample i', range: [0, N - 1] })
  const ty = useAxis({ label: 'value', key: state.signal })
  const ex = useAxis({ label: 'terms kept s', range: [0, N] })
  const ey = useAxis({ label: 'relative error', log: true, range: [1e-4, 1] })
  const ax = useAxis({ label: 'sample i', range: [0, N - 1] })
  const ay = useAxis({
    label: 'atoms used, largest coefficient first',
    key: `${state.signal}/${state.basis}/${atoms.length}`,
  })
  return (
    <Figure
      title="The same signal is sparse in one basis and dense in another"
      state={state}
      caption="Left: a signal of 64 samples (grey) and its best approximation with s terms of the chosen basis, the s coefficients of largest magnitude. Right: the relative error of the best s-term approximation in each orthonormal basis, the root of the energy left in the dropped coefficients. Drag the vertical line to change s. Below: the atoms of the chosen basis that the approximation uses, each multiplied by its coefficient and scaled to fit its row, largest coefficient first (at most eight); the approximation on the left is the sum of all s kept atoms. The smooth signal needs few cosines, the steps few Haar wavelets, the spikes few spikes. No basis serves the smooth signal with spikes added: its error falls slowly in all three."
      readouts={
        <>
          {BASES.map((b, i) => (
            <Readout key={b.id} label={`error with ${state.s} ${b.label}`} value={formatNumber(curves[i][state.s])} />
          ))}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={tx} y={ty} height={260}>
          <Curve name="signal" x={T} y={x} muted width={3} />
          <Curve name={`${state.s}-term approximation`} x={T} y={approx} slot={shown.slot} width={1.5} />
          <Points name="samples" x={T} y={x} muted size={3} />
        </Plot>
        <Plot x={ex} y={ey} height={260}>
          {BASES.map((b, i) => (
            <Curve
              key={b.id}
              name={b.label}
              x={sAxis}
              y={curves[i].map((v) => Math.max(v, 1e-4))}
              slot={b.slot}
              width={b.id === state.basis ? 2.5 : 1.25}
            />
          ))}
          <Handle {...state.handle('s', { label: 'terms kept' })} />
        </Plot>
      </div>
      <Plot x={ax} y={ay} height={60 + 44 * atoms.length} legend={false}>
        {atoms.map((a) => (
          <Annotation key={`base-${a.j}`} y={a.offset} text={a.label} muted />
        ))}
        {atoms.map((a) => (
          <Curve key={a.j} name={a.label} x={T} y={a.y.map((v) => v + a.offset)} slot={shown.slot} width={1.5} />
        ))}
      </Plot>
    </Figure>
  )
}
