import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Raster,
  Curve,
  Points,
  Readout,
  useAxis,
} from 'aifn-render'
import { strokeGlyphs } from 'aifn-applied/data/synthetic'
import { pca } from 'aifn-applied/unsupervised/embedding/linear'
import { dataset } from 'aifn/learning/estimators'
import { nmfSteps, type NmfLoss, type NmfSolver } from 'aifn/numerics/factorisation'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'

const SOLVERS: { value: string; label: string; solver: NmfSolver; loss: NmfLoss }[] = [
  { value: 'mu-f', label: 'multiplicative, squared error', solver: 'multiplicative', loss: 'frobenius' },
  { value: 'mu-kl', label: 'multiplicative, KL divergence', solver: 'multiplicative', loss: 'kullback-leibler' },
  { value: 'hals', label: 'HALS (hierarchical ALS), squared error', solver: 'hals', loss: 'frobenius' },
]

/** A size x size image from a flat row, top row first, as Raster rows (row 0 at the bottom of the plot). */
const image = (v: ArrayLike<number>, offset: number, size: number) =>
  Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => v[offset + (size - 1 - r) * size + c]),
  )

export function NmfGlyphsExplorer() {
  const n = 200
  const [p, setP] = useState(0.3)
  const [noise, setNoise] = useState(0.05)
  const [seed, setSeed] = useState(0)
  const [solverKey, setSolverKey] = useState('mu-f')
  const [rank, setRank] = useState(8)
  const [sweeps, setSweeps] = useState(150)
  const [step, setStep] = useState(150)

  const size = 7

  const run = useMemo(() => {
    const g = strokeGlyphs(stream(`glyphs-${seed}`), { n, p, noise, size })
    const s = SOLVERS.find((x) => x.value === solverKey) ?? SOLVERS[0]
    const tr = trace(
      nmfSteps(g.x, { rank, solver: s.solver, loss: s.loss, tolerance: 0 }),
      undefined,
      sweeps,
      { stream: stream(`nmf-${seed}`) },
    )
    const steps = tr.steps.map((st) => ({
      t: st.t,
      objective: st.objective,
      H: Float64Array.from(toFlat(st.H)),
    }))
    const P = pca({ components: rank }).fit(dataset(g.x)).components
    return {
      steps,
      examples: Array.from(toFlat(g.x)).slice(0, 6 * size * size),
      pca: Float64Array.from(toFlat(P)),
    }
  }, [n, p, noise, seed, solverKey, rank, sweeps])

  const currentStep = Math.min(step, run.steps.length - 1)
  const now = run.steps[currentStep]

  const cells = Array.from({ length: size }, (_, i) => i)
  const px = useAxis({ label: '', range: [-0.5, size - 0.5], nice: false })
  const py = useAxis({ label: '', range: [-0.5, size - 0.5], nice: false, equal: px })
  const sweepAxis = useAxis({ label: 'sweep', range: [0, Math.max(1, sweeps)] })
  const objAxis = useAxis({
    label: 'objective',
    log: true,
    hold: 'union',
    key: `${n}-${p}-${noise}-${seed}-${solverKey}-${rank}`,
  })

  const partMax = Math.max(1e-9, ...now.H)
  const pcaMax = Math.max(1e-9, ...Array.from(run.pca, Math.abs))
  const shown = Math.min(rank, 8)

  return (
    <Figure
      title="NMF recovers additive parts (strokes); PCA learns signed global blends"
      purpose="Non-negativity prevents cancellation: because each glyph must be reconstructed purely by addition, NMF decomposes images into constituent strokes; PCA's orthogonal eigenvectors allow arbitrary positive and negative weights, producing global signed patterns."
      defaultSize="XL"
      controls={
        <>
          <ControlGroup title="1 · Data generation">
            <NumberSelector
              label="stroke probability p"
              value={p}
              onChange={setP}
              min={0.1}
              max={0.7}
              step={0.05}
              suggestions={[0.2, 0.3, 0.4, 0.5]}
            />
            <NumberSelector
              label="noise sd"
              value={noise}
              onChange={setNoise}
              min={0}
              max={0.2}
              step={0.02}
              suggestions={[0, 0.05, 0.1]}
            />
            <NumberSelector
              label="seed"
              value={seed}
              onChange={setSeed}
              min={0}
              max={20}
              step={1}
              suggestions={[0, 1, 2, 5]}
            />
          </ControlGroup>
          <ControlGroup title="2 · Factorisation model">
            <Select
              label="solver algorithm"
              value={solverKey}
              onChange={setSolverKey}
              options={SOLVERS.map((s) => ({ value: s.value, label: s.label }))}
            />
            <NumberSelector
              label="rank k (number of parts)"
              value={rank}
              onChange={setRank}
              min={2}
              max={16}
              step={1}
              suggestions={[4, 6, 8, 12]}
            />
            <NumberSelector
              label="sweeps"
              value={sweeps}
              onChange={(s) => {
                setSweeps(s)
                setStep(s)
              }}
              min={20}
              max={500}
              step={20}
              suggestions={[50, 100, 150, 300]}
            />
          </ControlGroup>
          <ControlGroup title="3 · Sweep progression">
            <Player
              value={currentStep}
              onChange={setStep}
              count={run.steps.length}
              label="sweep"
            />
          </ControlGroup>
        </>
      }
      readouts={
        <>
          <Readout label="sweep" value={`${now.t} of ${run.steps.length - 1}`} />
          <Readout label="objective value" value={now.objective.toFixed(4)} />
          <Readout label="basis rank k" value={rank} />
          <Readout label="solver" value={solverKey} />
        </>
      }
      caption="Row 1: Six sample synthetic glyphs composed of horizontal, vertical, and diagonal strokes on a 7 × 7 pixel grid. Row 2: Basis parts (rows of matrix H) recovered by NMF at the active sweep; play from sweep 0 to observe random diffuse noise sharpen cleanly into isolated stroke primitives. Row 3: Principal components from PCA (red: positive, blue: negative), which overlay opposing strokes to minimise global squared error. Bottom: Monotonically decreasing factorisation loss over sweeps."
    >
      <Plots rows={3} cols={8} heights={[1, 1, 1]}>
        {Array.from({ length: 8 }, (_, k) => (
          <Plot key={`g${k}`} x={px} y={py} bare title={k === 0 ? 'glyph 1' : k < 6 ? `glyph ${k + 1}` : undefined}>
            {k < 6 && (
              <Raster
                x={cells}
                y={cells}
                z={image(run.examples, k * size * size, size)}
                range={[0, 1]}
                colorBar={false}
              />
            )}
          </Plot>
        ))}
        {Array.from({ length: 8 }, (_, k) => (
          <Plot key={`h${k}`} x={px} y={py} bare title={k < shown ? `NMF part ${k + 1}` : undefined}>
            {k < shown && (
              <Raster
                x={cells}
                y={cells}
                z={image(
                  now.H.map((v) => v / partMax),
                  k * size * size,
                  size,
                )}
                range={[0, 1]}
                colorBar={false}
              />
            )}
          </Plot>
        ))}
        {Array.from({ length: 8 }, (_, k) => (
          <Plot key={`p${k}`} x={px} y={py} bare title={k < shown ? `PC ${k + 1}` : undefined}>
            {k < shown && (
              <Raster
                x={cells}
                y={cells}
                z={image(run.pca, k * size * size, size)}
                scale="diverging"
                range={[-pcaMax, pcaMax]}
                colorBar={false}
              />
            )}
          </Plot>
        ))}
      </Plots>
      <Plots rows={1}>
        <Plot x={sweepAxis} y={objAxis} title="Optimisation objective across sweeps">
          <Curve
            name="objective"
            x={run.steps.map((s) => s.t)}
            y={run.steps.map((s) => s.objective)}
            slot={0}
          />
          <Points name="current sweep" x={[now.t]} y={[now.objective]} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
