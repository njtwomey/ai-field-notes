import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Curve,
  Readout,
  useAxis,
} from 'aifn-render'
import { cocktailParty } from 'aifn-applied/data/synthetic'
import { fastIca, pca } from 'aifn-applied/unsupervised/embedding/linear'
import { dataset } from 'aifn/learning/estimators'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'

const column = (v: ArrayLike<number>, n: number, d: number, c: number) =>
  Array.from({ length: n }, (_, i) => v[i * d + c])

const standardise = (a: number[]) => {
  const m = a.reduce((s, u) => s + u, 0) / a.length
  const sd = Math.sqrt(a.reduce((s, u) => s + (u - m) ** 2, 0) / a.length) || 1
  return a.map((u) => (u - m) / sd)
}

const correlation = (a: number[], b: number[]) => {
  const x = standardise(a)
  const y = standardise(b)
  return x.reduce((s, u, i) => s + u * y[i], 0) / x.length
}

function align(estimates: number[][], truth: number[][]) {
  const used = new Set<number>()
  const out: { source: number; y: number[]; r: number }[] = []
  const order = estimates.map((e, k) => ({
    k,
    best: Math.max(...truth.map((t) => Math.abs(correlation(e, t)))),
  }))
  order.sort((a, b) => b.best - a.best)
  for (const { k } of order) {
    let best = -1
    let r = 0
    truth.forEach((t, j) => {
      if (used.has(j)) return
      const c = correlation(estimates[k], t)
      if (best < 0 || Math.abs(c) > Math.abs(r)) {
        best = j
        r = c
      }
    })
    used.add(best)
    out.push({
      source: best,
      y: standardise(estimates[k]).map((u) => (r < 0 ? -u : u)),
      r: Math.abs(r),
    })
  }
  return out.sort((a, b) => a.source - b.source)
}

const SOURCE_NAMES = ['sine wave', 'square wave', 'sawtooth']

export function CocktailPartyExplorer() {
  const [n, setN] = useState(400)
  const [noise, setNoise] = useState(0.0)
  const [seed, setSeed] = useState(0)
  const [step, setStep] = useState(25)
  const maxIterations = 30

  const run = useMemo(() => {
    const party = cocktailParty(stream(`cocktail-${seed}`), { n, noise })
    const x = dataset(party.mixed)
    const truth = [0, 1, 2].map((c) => column(toFlat(party.sources), n, 3, c))

    const steps = Array.from({ length: maxIterations + 1 }, (_, t) => {
      const m = fastIca({ maxSteps: t, tolerance: 0 }).fit(x, { stream: stream(`ica-${seed}`) })
      const s = toFlat(m.transform(party.mixed))
      return {
        aligned: align(
          [0, 1, 2].map((c) => column(s, n, 3, c)),
          truth,
        ),
        change: m.training.final.change,
      }
    })

    const p = toFlat(pca({ components: 3 }).fit(x).transform(party.mixed))
    const principal = align(
      [0, 1, 2].map((c) => column(p, n, 3, c)),
      truth,
    )

    return {
      t: Array.from(toFlat(party.t)),
      mixed: [0, 1, 2].map((c) => column(toFlat(party.mixed), n, 3, c)),
      truth: truth.map(standardise),
      steps,
      principal,
    }
  }, [n, noise, seed])

  const currentStep = Math.min(step, run.steps.length - 1)
  const now = run.steps[currentStep]

  const tAxis = useAxis({ label: 'time t', range: [0, 8] })
  const micAxis = useAxis({ label: 'microphones x(t)', hold: 'union', key: `${n}-${noise}-${seed}` })
  const srcAxis = useAxis({ label: 'FastICA y(t)', range: [-2.6, 2.6] })
  const pcAxis = useAxis({ label: 'PCA z(t)', range: [-2.6, 2.6] })

  return (
    <Figure
      title="FastICA unmixing a cocktail party: independent components vs PCA"
      purpose="Linear mixtures of independent non-Gaussian signals are more Gaussian than the sources by the central limit theorem; FastICA rotates whitened signals until non-Gaussianity is maximised, recovering the true sources, whereas PCA only decorrelates."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="1 · Signal and noise setup">
            <NumberSelector
              label="sample count n"
              value={n}
              onChange={setN}
              min={100}
              max={1500}
              step={50}
              suggestions={[200, 400, 800, 1200]}
            />
            <NumberSelector
              label="sensor noise sd"
              value={noise}
              onChange={setNoise}
              min={0}
              max={0.4}
              step={0.02}
              suggestions={[0, 0.05, 0.1, 0.2]}
            />
            <NumberSelector
              label="mixing seed"
              value={seed}
              onChange={setSeed}
              min={0}
              max={20}
              step={1}
              suggestions={[0, 1, 2, 3, 5]}
            />
          </ControlGroup>
          <ControlGroup title="2 · FastICA fixed-point iteration">
            <Player
              value={currentStep}
              onChange={setStep}
              count={run.steps.length}
              label="iteration"
            />
          </ControlGroup>
        </>
      }
      readouts={
        <>
          <Readout label="FastICA iteration" value={`${currentStep} of ${maxIterations}`} />
          {now.aligned.map((a) => (
            <Readout
              key={`ica-${a.source}`}
              label={`FastICA |corr| with ${SOURCE_NAMES[a.source]}`}
              value={a.r.toFixed(3)}
            />
          ))}
          {run.principal.map((a) => (
            <Readout
              key={`pca-${a.source}`}
              label={`PCA |corr| with ${SOURCE_NAMES[a.source]}`}
              value={a.r.toFixed(3)}
            />
          ))}
        </>
      }
      caption="Top panel: 3 microphone signals x₁(t), x₂(t), x₃(t) recording arbitrary linear combinations of 3 independent sources (sine, square, sawtooth) with optional sensor noise. Middle panel: FastICA outputs y(t) (solid) overlaid on true sources (dashed); play the iteration slider to watch the unmixing matrix rotate from a random starting orientation to isolate each voice. Bottom panel: PCA outputs z(t), which remain linear cross-talk mixtures because orthogonal variance alignment cannot unmix non-Gaussian sources."
    >
      <Plots rows={3}>
        <Plot x={tAxis} y={micAxis} title="Microphone recordings">
          {run.mixed.map((m, k) => (
            <Curve key={k} name={`microphone ${k + 1}`} x={run.t} y={m} slot={k + 3} width={1.5} />
          ))}
        </Plot>
        <Plot x={tAxis} y={srcAxis} title="FastICA recovered sources (dashed: ground truth)">
          {run.truth.map((s, k) => (
            <Curve key={`t${k}`} name={SOURCE_NAMES[k]} x={run.t} y={s} slot={k} dashed width={1} />
          ))}
          {now.aligned.map((a) => (
            <Curve key={a.source} name={`FastICA ${SOURCE_NAMES[a.source]}`} x={run.t} y={a.y} slot={a.source} width={1.5} />
          ))}
        </Plot>
        <Plot x={tAxis} y={pcAxis} title="PCA principal components">
          {run.principal.map((a) => (
            <Curve
              key={a.source}
              name={`PC ${a.source + 1}`}
              x={run.t}
              y={a.y}
              slot={a.source}
              width={1.5}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
