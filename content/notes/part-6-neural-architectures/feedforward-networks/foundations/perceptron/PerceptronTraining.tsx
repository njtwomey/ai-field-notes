import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

type Vec3 = [number, number, number]
const BOX = 1.2
const N = 40
const MAX_EPOCHS = 200

/** Points uniform in [-1, 1]², labelled by a fixed line, with those closer than `gap` to the line removed. */
function makeData(seed: number, gap: number) {
  const r = rng(seed)
  const angle = 2 * Math.PI * r.uniform()
  const truth: Vec3 = [Math.cos(angle), Math.sin(angle), 0.4 * r.uniform() - 0.2]
  const xs: Vec3[] = []
  const ys: number[] = []
  while (xs.length < N) {
    const x: Vec3 = [2 * r.uniform() - 1, 2 * r.uniform() - 1, 1]
    const s = truth[0] * x[0] + truth[1] * x[1] + truth[2]
    if (Math.abs(s) < gap) continue
    xs.push(x)
    ys.push(s > 0 ? 1 : -1)
  }
  return { xs, ys, truth }
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

/** Cyclic perceptron from w = 0. Returns every weight vector after each update. */
function train(xs: Vec3[], ys: number[]): Vec3[] {
  let w: Vec3 = [0, 0, 0]
  const history: Vec3[] = [w]
  for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
    let mistakes = 0
    xs.forEach((x, i) => {
      if (ys[i] * dot(w, x) <= 0) {
        w = [w[0] + ys[i] * x[0], w[1] + ys[i] * x[1], w[2] + ys[i] * x[2]]
        history.push(w)
        mistakes++
      }
    })
    if (mistakes === 0) break
  }
  return history
}

/** The part of the line w1·x + w2·y + w3 = 0 inside the plotted box, as two endpoints. */
function lineInBox([a, b, c]: Vec3): { x: number[]; y: number[] } {
  if (Math.abs(a) < 1e-12 && Math.abs(b) < 1e-12) return { x: [], y: [] }
  if (Math.abs(b) > Math.abs(a)) return { x: [-BOX, BOX], y: [-BOX, BOX].map((x) => -(a * x + c) / b) }
  return { x: [-BOX, BOX].map((y) => -(b * y + c) / a), y: [-BOX, BOX] }
}

export function PerceptronTraining() {
  const [seed, setSeed] = useState(3)
  const [gap, setGap] = useState(0.1)
  const [shown, setShown] = useState<number | null>(null)

  const { xs, ys, truth, history, bound, gamma } = useMemo(() => {
    const d = makeData(seed, gap)
    const norm = Math.hypot(...d.truth)
    const u: Vec3 = [d.truth[0] / norm, d.truth[1] / norm, d.truth[2] / norm]
    const g = Math.min(...d.xs.map((x, i) => d.ys[i] * dot(u, x)))
    const R = Math.max(...d.xs.map((x) => Math.hypot(...x)))
    return { ...d, history: train(d.xs, d.ys), gamma: g, bound: (R / g) ** 2 }
  }, [seed, gap])

  const updates = history.length - 1
  const step = Math.min(shown ?? updates, updates)
  const w = history[step]
  const mistakes = xs.filter((x, i) => ys[i] * dot(w, x) <= 0).length

  const series = useMemo((): XYSeries[] => {
    const current = lineInBox(w)
    const reference = lineInBox(truth)
    return [
      {
        name: 'points',
        type: 'scatter',
        x: xs.map((x) => x[0]),
        y: xs.map((x) => x[1]),
        group: ys.map((y) => (y > 0 ? 1 : 0)),
        groupNames: ['y = −1', 'y = +1'],
      },
      { name: 'labelling line', type: 'line', ...reference, muted: true, dashed: true },
      { name: `boundary after ${step} updates`, type: 'line', ...current, emphasis: true },
    ]
  }, [xs, ys, truth, w, step])

  return (
    <Interactive
      title="Perceptron updates until every point is on the right side"
      caption="Forty points are labelled by the dashed line; points closer to it than the gap are removed. The perceptron starts from w = 0 and cycles through the data, adding y·x to w at each mistake. Scrub through the updates to watch the boundary move. Shrinking the gap shrinks the margin γ, and the number of updates grows, always below the bound (R/γ)²."
      controls={
        <>
          <ParamSlider
            label="gap around the labelling line"
            value={gap}
            onChange={(v) => {
              setGap(v)
              setShown(null)
            }}
            min={0.01}
            max={0.4}
            step={0.01}
          />
          <ParamSlider
            label="update shown"
            value={step}
            onChange={setShown}
            min={0}
            max={Math.max(updates, 1)}
            step={1}
            withArrows
          />
          <div className="flex items-end">
            <ParamButton
              onClick={() => {
                setSeed((s) => s + 1)
                setShown(null)
              }}
            >
              New data
            </ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="updates to converge" value={updates} />
          <Readout label="mistakes at this update" value={mistakes} />
          <Readout label="margin γ of the labelling line" value={formatNumber(gamma)} />
          <Readout label="bound (R/γ)²" value={formatNumber(bound)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <XYChart series={series} xRange={[-BOX, BOX]} yRange={[-BOX, BOX]} equalAspect xLabel="x₁" yLabel="x₂" />
      </div>
    </Interactive>
  )
}
