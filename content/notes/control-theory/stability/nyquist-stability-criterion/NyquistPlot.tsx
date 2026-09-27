import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { carg, freqResponse, logspace, polyAdd, roots, type TF } from '../../_shared/control'

type Loop = { label: string; tf: TF; openLoopRhpPoles: number; initial: number }

const LOOPS = {
  lag: { label: 'K / (s+1)³', tf: { num: [1], den: [1, 3, 3, 1] }, openLoopRhpPoles: 0, initial: Math.log10(4) },
  unstable: {
    label: 'K / ((s−1)(s+2))',
    tf: { num: [1], den: [1, 1, -2] },
    openLoopRhpPoles: 1,
    initial: Math.log10(1),
  },
} satisfies Record<string, Loop>
type LoopKey = keyof typeof LOOPS

// Frequencies for ω > 0; the curve for ω < 0 is its mirror image in the real axis.
const W = logspace(-3, 3, 1200)

export function NyquistPlot() {
  const [key, setKey] = useState<LoopKey>('lag')
  const loop: Loop = LOOPS[key]
  const logK = useParam(loop.initial, { min: -1, max: 1.5, step: 0.01 })
  const k = 10 ** logK.value

  const view = useMemo(() => {
    const pos = W.map((w) => freqResponse(loop.tf, w))
    const re = pos.map((z) => k * z.re)
    const im = pos.map((z) => k * z.im)
    // Winding of 1 + L around the origin as ω runs from −∞ to +∞ (the large semicircle maps to L = 0).
    const path = [...re.map((r, i) => [r, -im[i]]).reverse(), ...re.map((r, i) => [r, im[i]])]
    let turn = 0
    for (let i = 1; i < path.length; i++) {
      const a = carg({ re: 1 + path[i - 1][0], im: path[i - 1][1] })
      const b = carg({ re: 1 + path[i][0], im: path[i][1] })
      let d = b - a
      d -= 2 * Math.PI * Math.round(d / (2 * Math.PI))
      turn += d
    }
    const clockwise = Math.round(-turn / (2 * Math.PI))
    const closed = roots(
      polyAdd(
        loop.tf.den,
        loop.tf.num.map((c) => k * c),
      ),
    )
    const rhp = closed.filter((z) => z.re > 0).length
    // A square window holding the curve and the critical point.
    const xs = [...re, -1, 0]
    const ys = [...im, ...im.map((v) => -v), 0]
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2
    const cy = 0
    const half = 1.15 * Math.max((Math.max(...xs) - Math.min(...xs)) / 2, Math.max(...ys.map(Math.abs)), 0.6)
    const series: XYSeries[] = [
      { name: 'L(iω), ω > 0', type: 'line', x: re, y: im, slot: 0 },
      { name: 'L(iω), ω < 0', type: 'line', x: re, y: im.map((v) => -v), slot: 0, dashed: true },
      { name: 'critical point −1', type: 'scatter', x: [-1], y: [0], emphasis: true },
    ]
    // Arrows showing the direction of travel for increasing ω, on both halves.
    const arrows: Segment[] = []
    for (const i of [Math.floor(W.length * 0.5), Math.floor(W.length * 0.58)]) {
      arrows.push({ from: [re[i], im[i]], to: [re[i + 8], im[i + 8]] })
      arrows.push({ from: [re[i + 8], -im[i + 8]], to: [re[i], -im[i]] })
    }
    return {
      series,
      arrows,
      clockwise,
      rhp,
      x: [cx - half, cx + half] as [number, number],
      y: [cy - half, cy + half] as [number, number],
    }
  }, [loop, k])

  const P = loop.openLoopRhpPoles
  return (
    <Interactive
      title="Counting encirclements of −1"
      caption="The Nyquist plot of the loop transfer function L(s) = K·G(s): the image of the imaginary axis, solid for positive frequencies and dashed for negative ones (a mirror image). Arrows show increasing ω. The closed loop has Z = N + P poles in the right half-plane, where N counts clockwise encirclements of −1 and P counts open-loop poles in the right half-plane. For K/(s+1)³ the curve crosses the negative real axis at −K/8, so K > 8 gives two clockwise encirclements. For K/((s−1)(s+2)), which is unstable on its own, the loop needs one anticlockwise encirclement, which happens once K > 2."
      controls={
        <>
          <ParamChoice
            label="loop"
            value={key}
            onChange={(v) => {
              setKey(v)
              logK.set(LOOPS[v].initial)
            }}
            options={(Object.keys(LOOPS) as LoopKey[]).map((l) => ({ value: l, label: LOOPS[l].label }))}
          />
          <ParamSlider label="gain K" param={logK} format={(v) => formatNumber(10 ** v)} />
        </>
      }
      readout={
        <>
          <Readout label="open-loop RHP poles P" value={P} />
          <Readout label="clockwise encirclements N" value={view.clockwise} />
          <Readout label="closed-loop RHP poles Z = N + P" value={view.clockwise + P} />
          <Readout label="check: roots of 1 + L in RHP" value={view.rhp} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          series={view.series}
          vectors={view.arrows}
          xLabel="Re L"
          yLabel="Im L"
          xRange={view.x}
          yRange={view.y}
          equalAspect
        />
      </div>
    </Interactive>
  )
}
