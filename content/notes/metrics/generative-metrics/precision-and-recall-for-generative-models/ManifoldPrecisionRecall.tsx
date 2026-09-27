import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

type Vec = [number, number]
const N = 150
const RANGE: [number, number] = [-5, 5]

const dist = (a: Vec, b: Vec) => Math.hypot(a[0] - b[0], a[1] - b[1])
/** Radius of each point's ball: the distance to its k-th nearest other point in the same set. */
const radii = (xs: Vec[], k: number) => xs.map((p) => xs.map((q) => dist(p, q)).sort((a, b) => a - b)[k])
/** Fraction of the query points that fall inside at least one ball of the reference set. */
const coverage = (queries: Vec[], ref: Vec[], r: number[]) =>
  queries.filter((q) => ref.some((p, i) => dist(q, p) <= r[i])).length / queries.length

/**
 * Improved precision and recall (Kynkäänniemi et al. 2019): each set's support is estimated as the union of balls
 * reaching each sample's k-th nearest neighbour. Precision is the share of generated samples inside the real support;
 * recall is the share of real samples inside the generated support.
 */
export function ManifoldPrecisionRecall() {
  const shift = useParam(0.5, { min: 0, max: 3, step: 0.1 })
  const spread = useParam(0.7, { min: 0.2, max: 1.8, step: 0.05 })
  const outliers = useParam(0, { min: 0, max: 20, step: 1 })
  const k = useParam(3, { min: 1, max: 10, step: 1 })

  const r = useMemo(() => {
    const g = rng(11)
    const real: Vec[] = Array.from({ length: N }, () => [g.normal(), g.normal()])
    const gen: Vec[] = Array.from({ length: N }, () => [
      shift.value + spread.value * g.normal(),
      spread.value * g.normal(),
    ])
    for (let i = 0; i < outliers.value; i++) gen[i] = [8 * g.uniform() - 4, 8 * g.uniform() - 4]
    const precision = coverage(gen, real, radii(real, k.value))
    const recall = coverage(real, gen, radii(gen, k.value))
    return { real, gen, precision, recall }
  }, [shift.value, spread.value, outliers.value, k.value])

  const pts = (name: string, p: Vec[], slot: number): XYSeries => ({
    name,
    type: 'scatter',
    x: p.map((v) => v[0]),
    y: p.map((v) => v[1]),
    slot,
  })

  return (
    <Interactive
      title="Precision and recall for a generator"
      caption="Real samples come from a standard Gaussian; generated samples from a shifted Gaussian with adjustable spread, plus optional uniformly scattered outliers. Precision measures fidelity (generated samples that look real); recall measures diversity (how much of the real distribution the generator covers). Shrinking the spread keeps precision high but lowers recall, like mode collapse. Adding a few outliers lowers precision but can raise recall, because each outlier's large ball covers real samples far from any typical generated point."
      controls={
        <>
          <ParamSlider label="generated shift" param={shift} />
          <ParamSlider label="generated spread" param={spread} />
          <ParamSlider label="generated outliers" param={outliers} format={(v) => String(v)} />
          <ParamSlider label="neighbours k" param={k} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="precision" value={formatNumber(r.precision)} />
          <Readout label="recall" value={formatNumber(r.recall)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart
          series={[pts('real', r.real, 0), pts('generated', r.gen, 1)]}
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
        />
      </div>
    </Interactive>
  )
}
