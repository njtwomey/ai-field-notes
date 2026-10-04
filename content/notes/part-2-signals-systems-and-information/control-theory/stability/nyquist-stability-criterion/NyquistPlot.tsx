import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Points,
  Readout,
  type Segment,
  useAxis,
  useFigureState,
  variants,
  Vectors,
} from 'aifn-render'
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
  // One gain per loop (a variants case each), so switching loop restores that loop's own gain.
  const gain = (l: LoopKey) => ({
    label: LOOPS[l].label,
    params: {
      logK: float(LOOPS[l].initial, {
        min: -1,
        max: 1.5,
        step: 0.01,
        label: 'gain K',
        points_per_decade: 2,
        logTransform: 'value-is-log',
        format: (v) => formatNumber(10 ** v),
      }),
    },
  })
  const state = useFigureState({
    loop: variants({ lag: gain('lag'), unstable: gain('unstable') }, { choiceLabel: 'loop' }),
  })
  const loop: Loop = LOOPS[state.loop.key]
  const k = 10 ** state.loop.values.logK

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
    const series = [
      { name: 'L(iω), ω > 0', x: re, y: im, slot: 0 },
      { name: 'L(iω), ω < 0', x: re, y: im.map((v) => -v), slot: 0, dashed: true },
      { name: 'critical point −1', x: [-1], y: [0], emphasis: true },
    ] as const
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
  const xAxis = useAxis({ label: 'Re L', range: view.x })
  const yAxis = useAxis({ label: 'Im L', range: view.y, equal: xAxis })
  return (
    <Figure
      title="Counting encirclements of −1"
      state={state}
      caption="The Nyquist plot of the loop transfer function L(s) = K·G(s): the image of the imaginary axis, solid for positive frequencies and dashed for negative ones (a mirror image). Arrows show increasing ω. The closed loop has Z = N + P poles in the right half-plane, where N counts clockwise encirclements of −1 and P counts open-loop poles in the right half-plane. For K/(s+1)³ the curve crosses the negative real axis at −K/8, so K > 8 gives two clockwise encirclements. For K/((s−1)(s+2)), which is unstable on its own, the loop needs one anticlockwise encirclement, which happens once K > 2."
      readouts={
        <>
          <Readout label="open-loop RHP poles P" value={P} />
          <Readout label="clockwise encirclements N" value={view.clockwise} />
          <Readout label="closed-loop RHP poles Z = N + P" value={view.clockwise + P} />
          <Readout label="check: roots of 1 + L in RHP" value={view.rhp} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...view.series[0]} />
          <Curve {...view.series[1]} />
          <Points {...view.series[2]} />
          <Vectors vectors={view.arrows} />
        </Plot>
      </div>
    </Figure>
  )
}
