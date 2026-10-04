import { useMemo, useState } from 'react'
import { Figure, ControlGroup, NumberSelector, Plots, Plot, Points, Bars, Handle, Readout, useAxis } from 'aifn-render'
import { dataset } from 'aifn/learning/estimators'
import { BORDER, CORE, dbscan, optics } from 'aifn-methods/unsupervised/clustering'
import { moons } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x0: rows.map((r) => r[0]), x1: rows.map((r) => r[1]) }
}

export function OpticsExplorer() {
  const data = useMemo(() => moons(stream('optics/moons'), { n: 200, noise: 0.08 }), [])
  const cols = useMemo(() => columns(data.x), [data])

  const [eps, setEps] = useState(0.15)
  const [minSamples, setMinSamples] = useState(5)

  const model = useMemo(() => dbscan({ eps, minSamples }).fit(dataset(data.x)), [data, eps, minSamples])
  const order = useMemo(() => optics({ minSamples }).fit(dataset(data.x)), [data, minSamples])

  const groups = useMemo(() => {
    const labels = toFlat(model.labels)
    const roles = toFlat(model.roles)
    const pick = (f: (i: number) => boolean) => {
      const idx = cols.x0.map((_, i) => i).filter(f)
      return { idx, x: idx.map((i) => cols.x0[i]), y: idx.map((i) => cols.x1[i]) }
    }
    const core = pick((i) => roles[i] === CORE)
    return {
      core: { ...core, group: core.idx.map((i) => labels[i] % 8) },
      border: (() => {
        const b = pick((i) => roles[i] === BORDER)
        return { ...b, ring: b.idx.map(() => 'rgba(0,0,0,0)') }
      })(),
      noise: pick((i) => labels[i] < 0),
    }
  }, [model, cols])

  const reach = useMemo(() => {
    const r = toFlat(order.reachability)
    const ordering = toFlat(order.ordering)
    const bars = ordering.map((p) => (Number.isFinite(r[p]) ? r[p] : NaN))
    const top = Math.max(...bars.filter(Number.isFinite)) * 1.1
    return {
      x: ordering.map((_, i) => i),
      y: bars.map((b) => (Number.isFinite(b) ? b : top)),
      top,
    }
  }, [order])

  const counts = {
    core: groups.core.idx.length,
    border: groups.border.idx.length,
    noise: groups.noise.idx.length,
  }

  const x0 = useAxis({ label: 'x₀' })
  const x1 = useAxis({ label: 'x₁', equal: x0 })
  const orderAxis = useAxis({ label: 'OPTICS ordering', range: [-0.5, reach.x.length - 0.5], nice: false })
  const reachAxis = useAxis({ label: 'reachability distance', range: [0, reach.top] })

  return (
    <Figure
      title="OPTICS reachability plot and DBSCAN extraction"
      purpose="OPTICS orders points along density paths so that cluster structures across all ε radii are visible in a single reachability diagram; dragging ε slices the valleys into clusters."
      defaultSize="L"
      controls={
        <ControlGroup title="Density parameters">
          <NumberSelector
            label="ε threshold"
            value={eps}
            onChange={(v) => setEps(Math.max(0.02, Math.min(0.5, v)))}
            min={0.02}
            max={0.5}
            step={0.01}
            suggestions={[0.08, 0.12, 0.15, 0.2, 0.3]}
          />
          <NumberSelector
            label="minSamples"
            value={minSamples}
            onChange={setMinSamples}
            min={2}
            max={15}
            step={1}
            suggestions={[3, 5, 8, 10]}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="clusters" value={model.clusters} />
          <Readout label="core points" value={counts.core} />
          <Readout label="border points" value={counts.border} />
          <Readout label="noise points" value={counts.noise} />
        </>
      }
      caption="Drag the horizontal ε line on the reachability diagram below: each valley below the dashed threshold defines a cluster, while high peaks mark boundaries between clusters or isolated noise points. In the top plot, core points are filled by cluster, border points are ringed, and noise is grey."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={x0} y={x1} legend={false}>
          <Points name="noise" x={groups.noise.x} y={groups.noise.y} muted />
          <Points
            name="core"
            x={groups.core.x}
            y={groups.core.y}
            group={groups.core.group}
            groupNames={Array.from({ length: 8 }, (_, c) => `cluster ${c + 1}`)}
          />
          <Points name="border" x={groups.border.x} y={groups.border.y} colors={groups.border.ring} size={10} />
        </Plot>
        <Plot x={orderAxis} y={reachAxis} legend={false}>
          <Bars name="reachability" x={reach.x} y={reach.y} />
          <Handle
            kind="y"
            at={eps}
            label={`ε = ${eps.toFixed(2)}`}
            onDrag={(y) => setEps(Math.max(0.02, Math.min(reach.top, y)))}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
