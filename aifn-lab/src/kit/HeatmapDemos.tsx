import { normal, stream } from 'aifn/random'
import { useMemo, useState } from 'react'
import { ChartSize, formatNumber, Heatmap, Readout, XYChart, type Handle, type Vec2, type XYSeries } from '@lab/viz'
import { Player, Select, Slider, Switch, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'

const grid = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

/** Three Gaussian bumps: their weighted density, its gradient, and which bump dominates where. */
const BUMPS = [
  { at: [-1.5, -1] as Vec2, weight: 1, width: 0.9 },
  { at: [1.6, -0.6] as Vec2, weight: 0.8, width: 0.7 },
  { at: [0.2, 1.7] as Vec2, weight: 0.6, width: 1.1 },
]
const bump = (b: (typeof BUMPS)[number], x: number, y: number) =>
  b.weight * Math.exp(-((x - b.at[0]) ** 2 + (y - b.at[1]) ** 2) / (2 * b.width ** 2))
const density = (x: number, y: number) => BUMPS.reduce((s, b) => s + bump(b, x, y), 0)
function gradient(x: number, y: number): Vec2 {
  let [gx, gy] = [0, 0]
  for (const b of BUMPS) {
    const v = bump(b, x, y) / b.width ** 2
    gx -= v * (x - b.at[0])
    gy -= v * (y - b.at[1])
  }
  return [gx, gy]
}

type Scale = 'sequential' | 'diverging' | 'categorical'

/** A heatmap with every overlay: contours, a gradient-ascent path from a draggable start, its gradient and a scale choice. */
export function HeatmapFigure() {
  const [scale, setScale] = useState<Scale>('sequential')
  const [contours, setContours] = useState(true)
  const [start, setStart] = useState<Vec2>([2.2, 2.4])
  const rate = useParam(0.6, { min: 0.05, max: 2 })
  const x = useMemo(() => grid(-3.5, 3.5, 71), [])
  const y = x
  const fields = useMemo(() => {
    const dens = y.map((yv) => x.map((xv) => density(xv, yv)))
    const mean = dens.flat().reduce((a, b) => a + b, 0) / dens.flat().length
    return {
      sequential: dens,
      diverging: dens.map((row) => row.map((v) => v - mean)),
      categorical: y.map((yv) =>
        x.map((xv) => {
          const v = BUMPS.map((b) => bump(b, xv, yv))
          const top = Math.max(...v)
          return top < 0.05 ? -1 : v.indexOf(top)
        }),
      ),
    }
  }, [x, y])
  const path = useMemo(() => {
    const px = [start[0]]
    const py = [start[1]]
    for (let i = 0; i < 40; i++) {
      const [gx, gy] = gradient(px[i], py[i])
      px.push(px[i] + rate.value * gx)
      py.push(py[i] + rate.value * gy)
    }
    return { x: px, y: py }
  }, [start, rate.value])
  const z = fields[scale]
  const signed = scale === 'diverging'
  const bound = signed ? Math.max(...z.flat().map(Math.abs)) : 0
  const handles = useMemo((): Handle[] => [{ kind: 'point', at: start, onDrag: setStart }], [start])
  const overlay = useMemo(() => [{ name: 'gradient ascent', type: 'line' as const, ...path, showPoints: true }], [path])
  const [gx, gy] = gradient(...start)
  const vectors = useMemo(() => [{ from: start, to: [start[0] + gx, start[1] + gy] as Vec2 }], [start, gx, gy])
  const levels = useMemo(() => ({ levels: [0.2, 0.4, 0.6, 0.8], field: fields.sequential }), [fields])
  return (
    <Figure
      title="Heatmap with overlays, contours and a draggable start"
      description="A mixture of three bumps on a 71 × 71 grid, as a density (sequential), centred (diverging) or by dominant bump (categorical)."
      defaultSize="L"
      controls={
        <>
          <Select
            label="scale"
            value={scale}
            onChange={setScale}
            options={[
              { value: 'sequential', label: 'sequential: density' },
              { value: 'diverging', label: 'diverging: density − mean' },
              { value: 'categorical', label: 'categorical: dominant bump' },
            ]}
          />
          <Slider label="step size" param={rate} />
          <Switch label="contours" checked={contours} onChange={setContours} />
        </>
      }
      readouts={
        <>
          <Readout label="density at start" value={formatNumber(density(...start))} />
          <Readout label="density at end" value={formatNumber(density(path.x[40], path.y[40]))} />
        </>
      }
      caption="Drag the start point anywhere on the grid: the path and the gradient arrow follow without redrawing the cells. Hover a cell for its value; zoom with the toolbar or pinch."
    >
      <Heatmap
        x={x}
        y={y}
        z={z}
        scale={scale}
        range={signed ? [-bound, bound] : undefined}
        categoryNames={['bump 1', 'bump 2', 'bump 3']}
        contours={contours ? levels : undefined}
        overlay={overlay}
        marker={start}
        vectors={vectors}
        handles={handles}
        xLabel="x₁"
        yLabel="x₂"
        valueLabel={scale === 'categorical' ? 'dominant' : 'density'}
        fillOpacity={0.9}
      />
    </Figure>
  )
}

/**
 * A trace-like figure: three panels linked by `hoverGroup`, each with the current step as a draggable line, played by a
 * Player. Hovering any panel moves the pointer in all three, and the readout lists every panel's values.
 */
export function LinkedFigure() {
  const trace = useMemo(() => {
    const s = stream('trace')
    const n = 300
    const step = Array.from({ length: n }, (_, i) => i)
    const w = [[2], [-1.5], [0.5]]
    const loss = [0]
    const ms: number[] = []
    for (let i = 0; i < n; i++) {
      const noise = s.child(i)
      if (i > 0)
        for (let k = 0; k < 3; k++) w[k].push(w[k][i - 1] - 0.03 * (k + 1) * w[k][i - 1] + 0.02 * normal(noise, 0, 1))
      loss[i] = w.reduce((acc, wk, k) => acc + (k + 1) * wk[i] ** 2, 0)
      ms.push(0.4 + 0.1 * Math.abs(normal(noise.child('t'), 0, 1)) + (i % 50 === 0 ? 1.2 : 0))
    }
    return { step, loss, w, ms }
  }, [])
  const [position, setPosition] = useState(80)
  const cursor = useMemo(
    (): Handle[] => [
      { kind: 'x', at: position, onDrag: (x: number) => setPosition(Math.round(Math.min(Math.max(x, 0), 299))) },
    ],
    [position],
  )
  const loss = useMemo(
    (): XYSeries[] => [{ name: 'loss', type: 'line', x: trace.step, y: trace.loss, slot: 0 }],
    [trace],
  )
  const weights = useMemo(
    (): XYSeries[] => trace.w.map((y, k) => ({ name: `w${'₁₂₃'[k]}`, type: 'line', x: trace.step, y, slot: k })),
    [trace],
  )
  const timing = useMemo(
    (): XYSeries[] => [{ name: 'ms per step', type: 'area', x: trace.step, y: trace.ms, slot: 4 }],
    [trace],
  )
  return (
    <Figure
      title="Linked hover across panels"
      description="Loss, weights and step time of a noisy descent, as a trace view would show them."
      defaultSize="L"
      controls={<Player className="col-span-full" value={position} onChange={setPosition} count={300} />}
      readouts={
        <>
          <Readout label="step" value={position} />
          <Readout label="loss" value={formatNumber(trace.loss[position])} />
        </>
      }
      caption="Hover any panel: the pointer moves in all three (they share a hoverGroup) and the readout above lists every panel's values at that step. Drag the dashed line in any panel, or use the player, to move the current step. Each panel is 45% of the frame's height (ChartSize)."
    >
      <ChartSize scale={0.45}>
        <div className="flex flex-col gap-2">
          <XYChart
            series={loss}
            xLabel="step"
            yLabel="loss"
            yLog
            hoverGroup="kit-trace"
            handles={cursor}
            zoom={false}
          />
          <XYChart series={weights} xLabel="step" yLabel="w" hoverGroup="kit-trace" handles={cursor} zoom={false} />
          <XYChart series={timing} xLabel="step" yLabel="ms" hoverGroup="kit-trace" handles={cursor} zoom={false} />
        </div>
      </ChartSize>
    </Figure>
  )
}
