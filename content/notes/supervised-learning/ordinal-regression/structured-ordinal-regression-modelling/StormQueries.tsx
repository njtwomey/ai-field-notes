import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  type Handle,
  type HeatmapOverlay,
} from 'aifn-render'
import { useClassColors } from '../_shared/classColor'
import { OrdinalDataControls } from '../_shared/OrdinalDataControls'
import { NYSTROM_LANDMARKS, ordinalMetrics, type Point } from '../_shared/ordinal'
import { useFit } from '../_shared/useFit'
import { useGrid, useOrdinalData } from '../_shared/useOrdinalData'

/** The chain of K − 1 bits, each shaded by P(bit n = 1 | x), with the input joined to every node and edge. */
function chain(marginals: number[]): DiagramSpec {
  const n = marginals.length
  return {
    nodes: [
      ...marginals.map((m, i) => variable(`y${i}`, 1 + 2 * i, 1, `$y_{${i + 1}}$`, { shade: m, w: 0.95, h: 0.95 })),
      variable('x', n, 3, '$\\xvec$', { filled: true, w: 0.95, h: 0.95 }),
    ],
    edges: [
      ...marginals.slice(1).map((_, i) => link(`y${i}`, `y${i + 1}`, false)),
      ...marginals.map((_, i) => link('x', `y${i}`, false, { dashed: true })),
    ],
    unit: 44,
  }
}

/**
 * StORM's versatile queries: an interval probability P(a ≤ y ≤ b | x) over the plane, read from the chain as the
 * probability that bit a − 1 is on and bit b is off, and the per-bit marginals at a draggable point.
 */
type Features = 'storm' | 'storm-poly2' | 'storm-poly3' | 'storm-nystrom'
const FEATURES = [
  { value: 'storm' as const, label: 'linear' },
  { value: 'storm-poly2' as const, label: 'polynomial, degree 2' },
  { value: 'storm-poly3' as const, label: 'polynomial, degree 3' },
  { value: 'storm-nystrom' as const, label: `Nyström, ${NYSTROM_LANDMARKS} landmarks` },
]

type View = 'valid' | 'chain' | 'invalid'
const VIEWS = [
  { value: 'valid' as const, label: 'P(a ≤ y ≤ b), valid codes' },
  { value: 'chain' as const, label: 'chain marginal' },
  { value: 'invalid' as const, label: 'invalid-code mass' },
]

export function StormQueries() {
  const [view, setView] = useState<View>('valid')
  const [features, setFeatures] = useState<Features>('storm-poly2')
  const { spec, setSpec, resolution, setResolution } = useOrdinalData({ shape: 'spiral', k: 10 })
  const [range, setRange] = useState<[number, number]>([3, 5])
  const [query, setQuery] = useState<Point>([0, 1.5])
  const { fitted, data: d, fitting } = useFit(spec, features)
  const grid = useGrid(d.range, resolution)
  const colors = useClassColors(d.k)
  const model = fitted.storm!
  const lo = Math.min(range[0], d.k)
  const hi = Math.min(Math.max(range[1], lo), d.k)

  const z = useMemo(() => {
    const at = (x: Point) =>
      view === 'valid'
        ? model.interval(x, lo - 1, hi - 1)
        : view === 'chain'
          ? model.bitInterval(x, lo - 1, hi - 1)
          : model.invalidMass(x)
    return grid.map((y) => grid.map((x) => at([x, y])))
  }, [model, lo, hi, grid, view])
  // Points in their class colours, as in every ordinal figure; the marker shape says whether the class is in [a, b].
  const overlay = useMemo<HeatmapOverlay[]>(
    () => [
      {
        name: 'training point',
        type: 'scatter',
        x: d.train.x.map((p) => p[0]),
        y: d.train.x.map((p) => p[1]),
        colors: d.train.y.map((y) => colors[y]),
        group: d.train.y.map((y) => (y >= lo - 1 && y <= hi - 1 ? 0 : 1)),
        groupNames: ['class in [a, b]', 'class outside [a, b]'],
      },
    ],
    [d, lo, hi, colors],
  )

  const marginals = model.bitMarginals(query)
  const classProbs = model.classProbs(query)
  const invalid = 1 - classProbs.reduce((a, b) => a + b, 0)
  const held = useMemo(
    () => ({
      mae: ordinalMetrics(d.test.y, d.test.x.map(fitted.predict), d.k).mae,
      invalid: d.test.x.reduce((s, p) => s + model.invalidMass(p), 0) / d.test.x.length,
    }),
    [d, fitted, model],
  )
  const handles: Handle[] = [{ kind: 'point', at: query, label: 'x', onDrag: setQuery }]

  return (
    <Interactive
      title="Querying a StORM chain"
      caption={
        <MathText text="The background shows one of three maps from a StORM fitted to the training points, with potentials linear in the inputs or in a polynomial or Nyström expansion of them. The default is $P(a \le y \le b \mid \xvec)$ over the valid codes: the probabilities of classes $a$ to $b$ divided by the total probability of all $K$ valid codes. The chain marginal $P(y_{a-1} = 1, y_b = 0 \mid \xvec)$ gives the same number where the chain puts its mass on valid codes, but it also counts invalid codes, and far from the data those take almost all the mass; the third map shows that invalid-code mass. The diagram shows the chain of up-to-$k$ bits at the query point, each shaded by $P(y_n = 1 \mid \xvec)$. Points are coloured by class; a circle marks a class inside $[a, b]$ and a square one outside it. Drag the query point; set the interval with the sliders." />
      }
      controls={
        <>
          <OrdinalDataControls spec={spec} setSpec={setSpec} resolution={resolution} setResolution={setResolution} />
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamChoice label="features" value={features} onChange={setFeatures} options={FEATURES} />
          </div>
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamChoice label="map" value={view} onChange={setView} options={VIEWS} />
          </div>
          <ParamSlider
            label="lowest class a"
            value={lo}
            onChange={(v) => setRange([v, Math.max(v, hi)])}
            min={1}
            max={d.k}
            step={1}
          />
          <ParamSlider
            label="highest class b"
            value={hi}
            onChange={(v) => setRange([Math.min(lo, v), v])}
            min={1}
            max={d.k}
            step={1}
          />
        </>
      }
      readout={
        <>
          {classProbs.map((p, k) => (
            <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
          ))}
          {fitting && <Readout label="fitting" value="…" />}
          <Readout label="held-out MAE" value={formatNumber(held.mae)} />
          <Readout label="mean invalid-code mass, held-out points" value={formatNumber(held.invalid)} />
          <Readout label="invalid codes at x" value={formatNumber(invalid)} />
          <Readout label={`P(${lo} ≤ y ≤ ${hi}) at x`} value={formatNumber(model.interval(query, lo - 1, hi - 1))} />
        </>
      }
    >
      <div className="grid grid-cols-1 items-center gap-4 lg:grid-cols-[3fr_2fr]">
        <Heatmap
          x={grid}
          y={grid}
          z={z}
          range={[0, 1]}
          overlay={overlay}
          handles={handles}
          xLabel="x₁"
          yLabel="x₂"
          valueLabel={view === 'invalid' ? 'invalid-code mass' : `P(${lo} ≤ y ≤ ${hi})`}
          height={360}
          ariaLabel="Interval probability over the input plane"
        />
        <Diagram spec={chain(marginals)} ariaLabel="Chain of up-to-k bits shaded by their marginal probabilities" />
      </div>
    </Interactive>
  )
}
