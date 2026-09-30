import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Vector,
  type XYSeries,
} from '@/components/viz'
import type { LossSurface } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { mean, sigmoid } from '@/lib/math'

type Vec = [number, number]

const penalty = (lambda: number, a: number, b: number) => 0.5 * lambda * (a * a + b * b)

/** Boundary w1·x1 + w2·x2 = 0 across the plotted x-range, or undefined when it is vertical. */
function boundary(w1: number, w2: number): number[] | undefined {
  return Math.abs(w2) > 1e-6 ? [-4, 4].map((x1) => (-w1 * x1) / w2) : undefined
}

/**
 * The unregularised surface comes from python/mlc/figures/logistic_regression.py. The L2 term, the current point
 * and the regularised minimum are computed here.
 */
export function LossSurfaceExplorer() {
  const { data, error } = useFigure<LossSurface>('logistic-regression/loss-surface')
  const w1Param = useParam(-2, { min: -4, max: 4, step: 0.1 })
  const w2Param = useParam(2.5, { min: -4, max: 4, step: 0.1 })
  const w1 = w1Param.value
  const w2 = w2Param.value
  const [lambda, setLambda] = useState(0)

  // Regularised surface and its minimum on the grid. Recomputed only when λ changes.
  const surface = useMemo(() => {
    if (!data) return undefined
    const { x, y, z } = data.surface
    const total = z.map((row, i) => row.map((v, j) => v + penalty(lambda, x[j], y[i])))
    let best = { i: 0, j: 0 }
    total.forEach((row, i) =>
      row.forEach((v, j) => {
        if (v < total[best.i][best.j]) best = { i, j }
      }),
    )
    return { z: total, minimum: [x[best.j], y[best.i]] as [number, number] }
  }, [data, lambda])

  const stats = useMemo(() => {
    if (!data) return undefined
    const { x, y, group } = data.data
    const labels = group ?? []
    const p = x.map((xi, i) => sigmoid(w1 * xi + w2 * y[i]))
    const crossEntropy = mean(
      p.map((pi, i) => -(labels[i] * Math.log(pi + 1e-12) + (1 - labels[i]) * Math.log(1 - pi + 1e-12))),
    )
    const accuracy = mean(p.map((pi, i) => ((pi >= 0.5 ? 1 : 0) === labels[i] ? 1 : 0)))
    // ∇ cross-entropy = Xᵀ(p − y)/n and ∇ penalty = λw.
    const grad: [number, number] = [
      mean(p.map((pi, i) => (pi - labels[i]) * x[i])),
      mean(p.map((pi, i) => (pi - labels[i]) * y[i])),
    ]
    return { crossEntropy, reg: penalty(lambda, w1, w2), accuracy, grad }
  }, [data, w1, w2, lambda])

  const overlay = useMemo(
    () =>
      surface
        ? [
            {
              name: 'minimum',
              type: 'scatter' as const,
              x: [surface.minimum[0]],
              y: [surface.minimum[1]],
              emphasis: true,
            },
          ]
        : [],
    [surface],
  )

  const series = useMemo((): XYSeries[] => {
    if (!data || !surface) return []
    const chosen = boundary(w1, w2)
    const optimal = boundary(...surface.minimum)
    return [
      { name: 'data', type: 'scatter', ...data.data, groupNames: data.data.group_names ?? undefined },
      ...(chosen ? [{ name: 'your boundary', type: 'line' as const, x: [-4, 4], y: chosen, slot: 3 }] : []),
      ...(optimal ? [{ name: 'minimum', type: 'line' as const, x: [-4, 4], y: optimal, slot: 4, dashed: true }] : []),
    ]
  }, [data, surface, w1, w2])

  // ∇ₓ(w·x) = w, so w is normal to the boundary and points toward y = 1. Drawn at unit length so the arrow shows
  // direction only; ‖w‖ is in the readout. It starts at the origin because the bias is fixed at 0.
  const normal = useMemo(() => {
    const norm = Math.hypot(w1, w2)
    // Coloured as class y = 1 (scatter slot 1), the class it points toward.
    return norm < 1e-9 ? [] : [{ from: [0, 0] as Vec, to: [w1 / norm, w2 / norm] as Vec, slot: 1 }]
  }, [w1, w2])

  // Negative gradients at w: the step that gradient descent with η = 1 would take, split into the part from the
  // cross-entropy and the part from the penalty. Slots start at 1 because slot 0 is the heatmap's own hue.
  const descent = useMemo((): Vector[] => {
    if (!stats) return []
    const [g1, g2] = stats.grad
    const from: Vec = [w1, w2]
    const out: Vector[] = [{ from, to: [w1 - g1, w2 - g2], slot: 1, label: '−∇ cross-entropy' }]
    if (lambda > 0) {
      out.push({ from, to: [w1 - lambda * w1, w2 - lambda * w2], slot: 2, label: '−∇ penalty' })
      out.push({ from, to: [w1 - g1 - lambda * w1, w2 - g2 - lambda * w2], label: '−∇ total' })
    }
    return out
  }, [stats, w1, w2, lambda])

  // On the loss surface, the weights themselves are the handle.
  const weightHandle: Handle[] = [
    {
      kind: 'point',
      at: [w1, w2],
      label: 'w',
      onDrag: ([a, b]) => {
        w1Param.set(a)
        w2Param.set(b)
      },
    },
  ]
  // In feature space, the tip of the unit normal turns w about the origin and keeps ‖w‖ (1 when w = 0).
  const normalHandle: Handle[] = [
    {
      kind: 'point',
      at: normal.length ? normal[0].to : [1, 0],
      label: 'direction of w',
      onDrag: ([a, b]) => {
        if (Math.hypot(a, b) < 1e-9) return
        const angle = Math.atan2(b, a)
        const norm = Math.hypot(w1, w2) || 1
        w1Param.set(norm * Math.cos(angle))
        w2Param.set(norm * Math.sin(angle))
      },
    },
  ]

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !surface || !stats) return null

  return (
    <Interactive
      title="Loss surface and decision boundary"
      caption="Left: mean cross-entropy plus the L2 penalty (λ/2)‖w‖² for every (w₁, w₂), bias fixed at 0. The diamond marks the minimum. The arrows at w are the negative gradients, the step gradient descent with η = 1 would take: from the cross-entropy, from the penalty (pointing at the origin), and their sum. At the minimum the first two are equal and opposite, so the sum vanishes. Right: the data, the boundary for your weights and the boundary at the minimum. The arrow is the unit normal w/‖w‖, coloured as class y = 1, toward which it points. The length ‖w‖, in the readout, sets how sharply P(y = 1) changes across the boundary. Drag the dot on the loss surface to move w, or drag the arrow tip to turn the boundary at fixed ‖w‖. Raise λ and the minimum moves toward the origin: a smaller ‖w‖, a softer boundary."
      controls={
        <>
          <ParamSlider label="w₁" param={w1Param} />
          <ParamSlider label="w₂" param={w2Param} />
          <ParamSlider label="L2 penalty λ" value={lambda} onChange={setLambda} min={0} max={1} step={0.01} />
        </>
      }
      readout={
        <>
          <Readout label="cross-entropy" value={formatNumber(stats.crossEntropy)} />
          <Readout label="penalty" value={formatNumber(stats.reg)} />
          <Readout label="total" value={formatNumber(stats.crossEntropy + stats.reg)} />
          <Readout
            label="‖∇ total‖"
            value={formatNumber(Math.hypot(stats.grad[0] + lambda * w1, stats.grad[1] + lambda * w2))}
          />
          <Readout label="‖w‖" value={formatNumber(Math.hypot(w1, w2))} />
          <Readout label="accuracy" value={`${(stats.accuracy * 100).toFixed(1)}%`} />
          <Readout label="minimum at" value={`(${surface.minimum.map(formatNumber).join(', ')})`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={data.surface.x}
          y={data.surface.y}
          z={surface.z}
          xLabel={data.surface.x_label}
          yLabel={data.surface.y_label}
          valueLabel="loss"
          overlay={overlay}
          handles={weightHandle}
          vectors={descent}
          range={[0, 3]}
          height={340}
        />
        <XYChart
          equalAspect
          xRange={[-4, 4]}
          yRange={[-4, 4]}
          xLabel="x₁"
          yLabel="x₂"
          series={series}
          vectors={normal}
          handles={normalHandle}
        />
      </div>
    </Interactive>
  )
}
