import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  T_END,
  forwardParticles,
  histogram,
  joinPaths,
  normals,
  reverseParticles,
  sampleMixture,
  vpMixture,
  type ReverseKind,
} from '../_shared/sde'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const STEPS = 250
const PARTICLES = 2000
/** Forward paths are simulated for the largest count once, so the slider only chooses how many to draw. */
const MAX_PATHS = 50
const H = T_END / STEPS
const XS = toFlat(linspace(-4.5, 4.5, 181))

/**
 * Forward noising of a two-mode density by dX = −½X dt + dW, and generation by the reverse-time SDE with the exact
 * score, started from N(0, 1) at t = 5. The lower panel compares the reverse particles at the chosen time with the
 * exact marginal p_t.
 */
export function ReverseDiffusion() {
  const state = useFigureState({
    kind: choice<Exclude<ReverseKind, 'ode'>>(
      [
        { value: 'sde', label: 'f − g² ∇log p_t' },
        { value: 'noscore', label: 'f only (no score)' },
      ],
      'sde',
      { label: 'reverse drift' },
    ),
    t: slider(0, T_END, 1, { step: H * 5, label: 'time t' }),
    count: int(14, { min: 1, max: MAX_PATHS, step: 1, label: 'paths', format: (v) => String(v) }),
  })

  const forward = useMemo(() => forwardParticles(sampleMixture(MAX_PATHS, 31), STEPS, 32), [])
  const reverse = useMemo(() => reverseParticles(state.kind, normals(PARTICLES, 33), STEPS, 34), [state.kind])

  const pathSeries = useMemo<SeriesSpec[]>(() => {
    const fTimes = forward.map((_, k) => k * H)
    const rTimes = reverse.map((_, k) => T_END - k * H)
    // Draw the first `count` of the simulated particles, so raising the count adds paths and keeps the others.
    const n = state.count
    const fw = joinPaths(Array.from({ length: n }, (_, i) => ({ x: fTimes, y: forward.map((s) => s[i]) })))
    const rv = joinPaths(Array.from({ length: n }, (_, i) => ({ x: rTimes, y: reverse.map((s) => s[i]) })))
    return [
      { name: 'forward: data → noise', type: 'line', ...fw, muted: true, thin: n > 1 },
      { name: 'reverse: noise → data', type: 'line', ...rv, slot: 0, thin: n > 1 },
    ]
  }, [forward, reverse, state.count])

  const k = Math.round((T_END - state.t) / H)
  const densitySeries = useMemo(() => {
    const hist = histogram(reverse[k], -4.5, 4.5, 60)
    return [
      { name: 'reverse particles', x: hist.x, y: hist.y, slot: 0 },
      { name: 'exact p_t', x: XS, y: XS.map((x) => vpMixture(x, state.t).p), emphasis: true },
    ] as const
  }, [reverse, k, state.t])

  const right = useMemo(() => reverse[k].reduce((s, x) => s + (x > -0.25 ? 1 : 0), 0) / PARTICLES, [reverse, k])
  const target = useMemo(() => {
    let s = 0
    for (const x of XS) if (x > -0.25) s += vpMixture(x, state.t).p * (XS[1] - XS[0])
    return s
  }, [state.t])

  const xAxis = useAxis({ label: 't', range: [0, T_END] })
  const yAxis = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const xAxis2 = useAxis({ label: 'x', range: [-4.5, 4.5] })
  const yAxis2 = useAxis({ label: 'density', range: [0, 0.9] })
  return (
    <Figure
      title="Running a diffusion backwards with the score"
      state={state}
      caption="Grey: the forward process dX = −½X dt + dW carries data from a two-mode density (35% near −2, 65% near 1.5) to N(0, 1) by t = 5. Colour: particles drawn from N(0, 1) at t = 5 and moved backwards by the reverse-time SDE, whose drift adds g²∇log p_t, with the exact score of the noised mixture. The paths slider sets how many paths of each kind are drawn, as light lines when there are several. The lower panel shows the reverse particles at the time on the slider against the exact p_t. Without the score term the reverse particles never find the data: their spread grows instead. Drag the vertical line to move in time."

      readouts={
        <>
          <Readout label="share of particles right of −0.25" value={formatNumber(right)} />
          <Readout label="same share under p_t" value={formatNumber(target)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        {seriesLayers(pathSeries)}
        <Handle {...state.handle('t', { label: 't' })} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={200}>
        <Bars {...densitySeries[0]} />
        <Curve {...densitySeries[1]} />
      </Plot>
    </Figure>
  )
}
