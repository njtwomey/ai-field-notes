import { useMemo, useState } from 'react'
import {
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Raster,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { forwardBackward, viterbi, type Hmm } from './hmm'

const START = [3, 6, 6, 6, 1, 6, 6, 2, 4, 1]
const STATES = [0, 1]
const fmt = (v: number) => v.toFixed(3)

/** Faces 1–6 from the fair die (state 0) or the loaded die (state 1), which favours six. */
function casino(toLoaded: number, toFair: number, six: number): Hmm {
  const stationaryLoaded = toLoaded / (toLoaded + toFair)
  return {
    initial: [1 - stationaryLoaded, stationaryLoaded],
    transition: [
      [1 - toLoaded, toLoaded],
      [toFair, 1 - toFair],
    ],
    emission: [Array(6).fill(1 / 6), [...Array(5).fill((1 - six) / 5), six]],
  }
}

/**
 * The trellis of a short dice sequence: posterior marginals p(y_n | x) as a heatmap with the Viterbi path on top,
 * filtering against smoothing, and the α, ψ, β vectors themselves. Click a roll to change its face.
 */
export function HmmTrellis() {
  const state = useFigureState({
    toLoaded: float(0.05, { min: 0.01, max: 0.5, step: 0.01, label: 'P(fair → loaded)' }),
    toFair: float(0.1, { min: 0.01, max: 0.5, step: 0.01, label: 'P(loaded → fair)' }),
    six: float(0.5, { min: 0.2, max: 0.95, step: 0.01, label: 'P(six | loaded)' }),
  })
  const [rolls, setRolls] = useState(START)

  const r = useMemo(() => {
    const m = casino(state.toLoaded, state.toFair, state.six)
    const xs = rolls.map((f) => f - 1)
    const fb = forwardBackward(m, xs)
    const vit = viterbi(m, xs)
    return { fb, vit }
  }, [state.toLoaded, state.toFair, state.six, rolls])

  const positions = rolls.map((_, n) => n + 1)
  // Heatmap rows are states (0 fair, 1 loaded), columns positions: z[state][n] = p(y_n = state | x).
  const z = STATES.map((v) => r.fb.marginal.map((p) => p[v]))
  const overlay = [{ name: 'Viterbi path', x: positions, y: r.vit.path, showPoints: true, emphasis: true }] as const
  const lines = [
    { name: 'filtering p(loaded | x₁…xₙ)', x: positions, y: r.fb.filtered.map((p) => p[1]), slot: 1 },
    { name: 'smoothing p(loaded | x)', x: positions, y: r.fb.marginal.map((p) => p[1]), slot: 0 },
  ] as const
  const cycle = (n: number) => setRolls((rs) => rs.map((f, i) => (i === n ? (f % 6) + 1 : f)))

  const xAxis = useAxis({ label: 'position n' })
  const yAxis = useAxis({ label: 'die (0 fair, 1 loaded)' })
  const xAxis2 = useAxis({ label: 'position n', hold: 'union' })
  const yAxis2 = useAxis({ label: 'p(loaded)', range: [0, 1] })
  return (
    <Figure
      title="Forward–backward and Viterbi on a short sequence"
      state={state}
      caption="Top: the posterior probability of each die at each roll (dark = probable), with the Viterbi path drawn on top. Middle: filtering uses only the rolls so far (α ⊙ ψ); smoothing also uses the rolls after (α ⊙ ψ ⊙ β), so it can revise an early roll once later rolls arrive. Bottom: the normalised vectors themselves. Click a roll to change its face, and move the sliders to change the model."

      readouts={
        <>
          <Readout label="log p(x)" value={formatNumber(r.fb.logZ)} />
          <Readout label="Viterbi log p(x, y*)" value={formatNumber(r.vit.logProbability)} />
          <Readout label="Viterbi path" value={r.vit.path.map((s) => (s ? 'L' : 'F')).join('')} />
          <Readout label="max-marginal path" value={r.fb.marginal.map((p) => (p[1] > 0.5 ? 'L' : 'F')).join('')} />
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-xs text-muted-foreground">rolls x₁ … x₁₀</span>
        {rolls.map((f, n) => (
          <Button key={n} variant="outline" size="icon-sm" onClick={() => cycle(n)} aria-label={`roll ${n + 1}: ${f}`}>
            {f}
          </Button>
        ))}
      </div>
      <Plot x={xAxis} y={yAxis} height={220}>
        <Raster x={positions} y={STATES} z={z} range={[0, 1]} valueLabel={'p(yₙ | x)'} />
        <Curve {...overlay[0]} live />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={220}>
        <Curve {...lines[0]} />
        <Curve {...lines[1]} />
      </Plot>
      <div className="overflow-x-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>n</TableHead>
              <TableHead>xₙ</TableHead>
              <TableHead>αₙ (fair, loaded)</TableHead>
              <TableHead>ψₙ</TableHead>
              <TableHead>βₙ</TableHead>
              <TableHead>p(yₙ | x)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rolls.map((f, n) => {
              const norm = (v: number[]) => {
                const s = v[0] + v[1]
                return `(${fmt(v[0] / s)}, ${fmt(v[1] / s)})`
              }
              return (
                <TableRow key={n}>
                  <TableCell className="font-mono">{n + 1}</TableCell>
                  <TableCell className="font-mono">{f}</TableCell>
                  <TableCell className="font-mono text-xs">{norm(r.fb.alpha[n])}</TableCell>
                  <TableCell className="font-mono text-xs">{`(${fmt(r.fb.psi[n][0])}, ${fmt(r.fb.psi[n][1])})`}</TableCell>
                  <TableCell className="font-mono text-xs">{norm(r.fb.beta[n])}</TableCell>
                  <TableCell className="font-mono text-xs">{`(${fmt(r.fb.marginal[n][0])}, ${fmt(r.fb.marginal[n][1])})`}</TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </Figure>
  )
}
