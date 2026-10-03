import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'
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

const STEPS = 250
const PARTICLES = 2000
/** Forward paths are simulated for the largest count once, so the slider only chooses how many to draw. */
const MAX_PATHS = 50
const H = T_END / STEPS
const XS = linspace(-4.5, 4.5, 181)

/**
 * Forward noising of a two-mode density by dX = −½X dt + dW, and generation by the reverse-time SDE with the exact
 * score, started from N(0, 1) at t = 5. The lower panel compares the reverse particles at the chosen time with the
 * exact marginal p_t.
 */
export function ReverseDiffusion() {
  const [kind, setKind] = useState<Exclude<ReverseKind, 'ode'>>('sde')
  const t = useParam(1, { min: 0, max: T_END, step: H * 5 })
  const count = useParam(14, { min: 1, max: MAX_PATHS, step: 1 })

  const forward = useMemo(() => forwardParticles(sampleMixture(MAX_PATHS, 31), STEPS, 32), [])
  const reverse = useMemo(() => reverseParticles(kind, normals(PARTICLES, 33), STEPS, 34), [kind])

  const pathSeries = useMemo<XYSeries[]>(() => {
    const fTimes = forward.map((_, k) => k * H)
    const rTimes = reverse.map((_, k) => T_END - k * H)
    // Draw the first `count` of the simulated particles, so raising the count adds paths and keeps the others.
    const n = count.value
    const fw = joinPaths(Array.from({ length: n }, (_, i) => ({ x: fTimes, y: forward.map((s) => s[i]) })))
    const rv = joinPaths(Array.from({ length: n }, (_, i) => ({ x: rTimes, y: reverse.map((s) => s[i]) })))
    return [
      { name: 'forward: data → noise', type: 'line', ...fw, muted: true, thin: n > 1 },
      { name: 'reverse: noise → data', type: 'line', ...rv, slot: 0, thin: n > 1 },
    ]
  }, [forward, reverse, count.value])

  const k = Math.round((T_END - t.value) / H)
  const densitySeries = useMemo<XYSeries[]>(() => {
    const hist = histogram(reverse[k], -4.5, 4.5, 60)
    return [
      { name: 'reverse particles', type: 'bar', x: hist.x, y: hist.y, slot: 0 },
      { name: 'exact p_t', type: 'line', x: XS, y: XS.map((x) => vpMixture(x, t.value).p), emphasis: true },
    ]
  }, [reverse, k, t.value])

  const right = useMemo(() => reverse[k].reduce((s, x) => s + (x > -0.25 ? 1 : 0), 0) / PARTICLES, [reverse, k])
  const target = useMemo(() => {
    let s = 0
    for (const x of XS) if (x > -0.25) s += vpMixture(x, t.value).p * (XS[1] - XS[0])
    return s
  }, [t.value])

  return (
    <Interactive
      title="Running a diffusion backwards with the score"
      caption="Grey: the forward process dX = −½X dt + dW carries data from a two-mode density (35% near −2, 65% near 1.5) to N(0, 1) by t = 5. Colour: particles drawn from N(0, 1) at t = 5 and moved backwards by the reverse-time SDE, whose drift adds g²∇log p_t, with the exact score of the noised mixture. The paths slider sets how many paths of each kind are drawn, as light lines when there are several. The lower panel shows the reverse particles at the time on the slider against the exact p_t. Without the score term the reverse particles never find the data: their spread grows instead. Drag the vertical line to move in time."
      controls={
        <>
          <ParamChoice
            label="reverse drift"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'sde', label: 'f − g² ∇log p_t' },
              { value: 'noscore', label: 'f only (no score)' },
            ]}
          />
          <ParamSlider label="time t" param={t} withArrows />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="share of particles right of −0.25" value={formatNumber(right)} />
          <Readout label="same share under p_t" value={formatNumber(target)} />
        </>
      }
    >
      <XYChart
        height={260}
        xLabel="t"
        yLabel="x"
        series={pathSeries}
        xRange={[0, T_END]}
        yRange={[-4.5, 4.5]}
        handles={[{ kind: 'x', at: t.value, label: 't', onDrag: (x) => t.set(x) }]}
      />
      <XYChart height={200} xLabel="x" yLabel="density" series={densitySeries} xRange={[-4.5, 4.5]} yRange={[0, 0.9]} />
    </Interactive>
  )
}
