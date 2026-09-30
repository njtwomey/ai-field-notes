import { useMemo, useState } from 'react'
import {
  describeRecipe,
  flippedMask,
  recipe,
  type ClassificationBase,
  type ClassificationTruth,
  type DatasetRecipe,
  type MissingMechanism,
} from 'aifn/datasets'
import { grid2d } from 'aifn/geometry'
import { toFlat } from 'aifn/tensor'
import { Button, Select, Slider, Switch } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type HeatmapOverlay, type XYSeries } from '@lab/viz'

// Every base is made binary here, so prevalence and the posterior P(y = 1 | x) apply to all of them.
const BASES: Record<
  ClassificationBase,
  {
    label: string
    knob?: { label: string; range: [number, number]; value: number }
    options?: DatasetRecipe['options']
  }
> = {
  moons: { label: 'moons', knob: { label: 'noise sd', range: [0.02, 0.6], value: 0.25 } },
  circles: { label: 'circles', knob: { label: 'noise sd', range: [0.02, 0.4], value: 0.12 } },
  blobs: { label: 'blobs', knob: { label: 'blob sd', range: [0.2, 3], value: 1 }, options: { centers: 2 } },
  gaussians: { label: 'Gaussians', knob: { label: 'class sd', range: [0.2, 3], value: 1 } },
  xor: { label: 'XOR (Gaussian)', knob: { label: 'blob sd', range: [0.1, 1.2], value: 0.5 } },
  spirals: { label: 'spirals', knob: { label: 'noise sd', range: [0.01, 0.2], value: 0.06 } },
  rings: { label: 'rings', knob: { label: 'radial sd', range: [0.02, 0.6], value: 0.2 }, options: { radii: [1, 2] } },
  checkerboard: { label: 'checkerboard', options: { tiles: 3 } },
}

const SEPARABLE = new Set<ClassificationBase>(['blobs', 'gaussians'])
const MECHANISMS: { value: MissingMechanism; label: string }[] = [
  { value: 'mcar', label: 'MCAR' },
  { value: 'mar', label: 'MAR (on x₁)' },
  { value: 'mnar', label: 'MNAR' },
]

const pct = (v: number) => `${Math.round(100 * v)}%`

export function RecipeSpecimen() {
  const [base, setBase] = useState<ClassificationBase>('moons')
  const [n, setN] = useState(400)
  const [knobs, setKnobs] = useState<Partial<Record<ClassificationBase, number>>>({})
  const [seed, setSeed] = useState(0)
  const [prevalence, setPrevalence] = useState(0.3)
  const [separation, setSeparation] = useState(2.5)
  const [labelNoise, setLabelNoise] = useState(0.05)
  const [outliers, setOutliers] = useState(0)
  const [nuisance, setNuisance] = useState(0)
  const [missing, setMissing] = useState(0)
  const [mechanism, setMechanism] = useState<MissingMechanism>('mcar')
  const [showTruth, setShowTruth] = useState(true)
  const b = BASES[base]
  const knob = b.knob ? (knobs[base] ?? b.knob.value) : undefined

  const spec = useMemo<DatasetRecipe>(
    () => ({
      base,
      seed,
      n,
      prevalence,
      ...(knob !== undefined && { noise: knob }),
      ...(SEPARABLE.has(base) && { separation }),
      ...(b.options && { options: b.options }),
      ...(labelNoise > 0 && { labelNoise }),
      ...(outliers > 0 && { outliers }),
      ...(nuisance > 0 && { nuisance }),
      ...(missing > 0 && { missing: { rate: missing, mechanism } }),
    }),
    [base, seed, n, prevalence, knob, separation, b, labelNoise, outliers, nuisance, missing, mechanism],
  )
  const data = useMemo(() => recipe(spec), [spec])
  const truth = data.meta.truth as ClassificationTruth | undefined
  const [rows, d] = data.x.shape

  // Points at their complete positions (before missing values), split into the groups the chart marks.
  const view = useMemo(() => {
    const full = toFlat(data.meta.complete ?? data.x)
    const mask = data.meta.missing ? toFlat(data.meta.missing) : undefined
    const y = toFlat(data.y!)
    const flipped = flippedMask(data)
    const px = Array.from({ length: rows }, (_, i) => full[i * d])
    const py = Array.from({ length: rows }, (_, i) => full[i * d + 1])
    const gappy = (i: number) => mask !== undefined && (mask[i * d] === 1 || mask[i * d + 1] === 1)
    const pick = (keep: (i: number) => boolean) => px.map((_, i) => i).filter(keep)
    const shown = pick((i) => !gappy(i))
    const flips = pick((i) => flipped[i] === 1)
    const holes = pick(gappy)
    const pad = (lo: number, hi: number): [number, number] => {
      const m = 0.08 * (hi - lo || 1)
      return [lo - m, hi + m]
    }
    const range = (v: number[]) => pad(Math.min(...v), Math.max(...v))
    return { px, py, y, shown, flips, holes, xr: range(px), yr: range(py), full }
  }, [data, rows, d])

  const field = useMemo(() => grid2d(view.xr, view.yr, 70), [view])
  const posterior = useMemo(() => {
    if (!truth || !showTruth) return undefined
    const pts = toFlat(field.points)
    const [ny, nx] = field.shape
    const probe = new Array<number>(d).fill(0)
    return Array.from({ length: ny }, (_, i) =>
      Array.from({ length: nx }, (_, j) => {
        probe[0] = pts[2 * (i * nx + j)]
        probe[1] = pts[2 * (i * nx + j) + 1]
        return truth.probability(probe)
      }),
    )
  }, [truth, showTruth, field, d])

  const names = useMemo(() => data.meta.labelNames ?? ['class 0', 'class 1'], [data])
  const at = (idx: number[]) => ({ x: idx.map((i) => view.px[i]), y: idx.map((i) => view.py[i]) })
  const overlay: HeatmapOverlay[] = [
    { name: 'points', type: 'scatter', ...at(view.shown), group: view.shown.map((i) => view.y[i]), groupNames: names },
    ...(view.flips.length
      ? [{ name: 'label flipped', type: 'scatter' as const, ...at(view.flips), emphasis: true }]
      : []),
    ...(view.holes.length ? [{ name: 'x₁ or x₂ missing', type: 'scatter' as const, ...at(view.holes), slot: 4 }] : []),
  ]
  const axisKey = `${base}:${n}:${seed}:${outliers > 0}`

  // Side panels: a nuisance feature against x₁, and the missingness mask.
  const pair: XYSeries[] = useMemo(() => {
    if (nuisance === 0) return []
    const extra = view.px.map((_, i) => view.full[i * d + d - nuisance])
    return [{ name: 'points', type: 'scatter', x: view.px, y: extra, group: view.y, groupNames: names }]
  }, [nuisance, view, d, names])
  const maskRows = useMemo(() => {
    if (!data.meta.missing) return undefined
    const m = toFlat(data.meta.missing)
    return Array.from({ length: rows }, (_, i) => m.slice(i * d, (i + 1) * d))
  }, [data, rows, d])

  const counts = [0, 1].map((j) => view.y.filter((v) => v === j).length)
  const flippedCount = view.flips.length
  const panels = 1 + (pair.length ? 1 : 0) + (maskRows ? 1 : 0)

  const main = posterior ? (
    <Heatmap
      x={toFlat(field.x)}
      y={toFlat(field.y)}
      z={posterior}
      scale="diverging"
      range={[0, 1]}
      fillOpacity={0.55}
      contours={{ levels: [0.5] }}
      overlay={overlay}
      valueLabel="P(y = 1 | x)"
      xLabel={data.meta.featureNames[0]}
      yLabel={data.meta.featureNames[1]}
      axisKey={axisKey}
    />
  ) : (
    <XYChart
      series={overlay.map((o): XYSeries => ({
        name: o.name,
        type: 'scatter',
        x: Array.from(o.x),
        y: Array.from(o.y),
        group: o.group ? Array.from(o.group) : undefined,
        groupNames: o.groupNames,
        emphasis: o.emphasis,
        slot: o.slot,
      }))}
      xLabel={data.meta.featureNames[0]}
      yLabel={data.meta.featureNames[1]}
      aspect="equal"
      axisKey={axisKey}
    />
  )

  return (
    <Figure
      title="Dataset recipe"
      description="A dataset built from plain parameters: a base generator, its class balance and overlap, label noise and extras. Where the process has a closed form the Bayes posterior P(y = 1 | x) is known, so the Bayes-optimal boundary (the 0.5 contour) and the Bayes error come with the data."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · base">
            <Select
              label="generator"
              value={base}
              onChange={setBase}
              options={(Object.keys(BASES) as ClassificationBase[]).map((k) => ({ value: k, label: BASES[k].label }))}
            />
            <Slider label="points n" value={n} onChange={(v) => setN(Math.round(v))} min={50} max={2000} step={10} />
            {b.knob && knob !== undefined && (
              <Slider
                label={b.knob.label}
                value={knob}
                onChange={(v) => setKnobs((m) => ({ ...m, [base]: v }))}
                min={b.knob.range[0]}
                max={b.knob.range[1]}
              />
            )}
            <Button variant="outline" size="sm" onClick={() => setSeed((s) => s + 1)}>
              new seed
            </Button>
          </ControlRow>
          <ControlRow label="2 · classes">
            <Slider label="prevalence of class 1" value={prevalence} onChange={setPrevalence} min={0.05} max={0.95} />
            <Slider
              label="separation d′"
              value={separation}
              onChange={setSeparation}
              min={0}
              max={6}
              disabled={!SEPARABLE.has(base)}
            />
            <Slider label="label noise" value={labelNoise} onChange={setLabelNoise} min={0} max={0.45} />
          </ControlRow>
          <ControlRow label="3 · extras">
            <Slider label="outliers" value={outliers} onChange={setOutliers} min={0} max={0.2} />
            <Slider
              label="nuisance features"
              value={nuisance}
              onChange={(v) => setNuisance(Math.round(v))}
              min={0}
              max={5}
              step={1}
            />
            <Slider label="missing rate" value={missing} onChange={setMissing} min={0} max={0.5} />
            <Select label="mechanism" value={mechanism} onChange={setMechanism} options={MECHANISMS} />
          </ControlRow>
          <ControlRow label="4 · show">
            <Switch label="Bayes posterior and boundary" checked={showTruth} onChange={setShowTruth} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label={names[0]} value={counts[0]} />
          <Readout label={names[1]} value={`${counts[1]} (${pct(counts[1] / rows)})`} />
          <Readout label="flipped" value={flippedCount} />
          <Readout label="features" value={d} />
          <Readout
            label="Bayes error"
            value={
              truth
                ? `${truth.bayesError.toFixed(3)}${truth.bayesErrorMethod === 'monte carlo' ? ` ± ${truth.bayesErrorSe.toFixed(3)} (MC)` : ' (exact)'}`
                : '—'
            }
          />
        </>
      }
      caption={`${describeRecipe(spec)}. ${data.meta.description} Ink diamonds mark flipped labels.${nuisance ? ' The pair plot shows x₁ against the last nuisance feature, which carries no information about y.' : ''}${maskRows ? ' The mask shows missing entries (rows × features).' : ''}`}
    >
      <Subplots cols={panels} widthRatios={panels > 1 ? [2, ...new Array<number>(panels - 1).fill(1)] : undefined}>
        <Panel>{main}</Panel>
        {pair.length > 0 && (
          <Panel>
            <XYChart
              series={pair}
              xLabel={data.meta.featureNames[0]}
              yLabel={data.meta.featureNames[d - 1]}
              axisKey={axisKey}
            />
          </Panel>
        )}
        {maskRows && (
          <Panel>
            <Heatmap
              x={Array.from({ length: d }, (_, j) => j + 1)}
              y={Array.from({ length: rows }, (_, i) => i)}
              z={maskRows}
              range={[0, 1]}
              colorBar={false}
              valueLabel="missing"
              xLabel="feature"
              yLabel="row"
            />
          </Panel>
        )}
      </Subplots>
    </Figure>
  )
}
