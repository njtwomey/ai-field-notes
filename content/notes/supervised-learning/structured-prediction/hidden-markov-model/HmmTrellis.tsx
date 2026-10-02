import { useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type HeatmapOverlay,
  type XYSeries,
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
  const toLoaded = useParam(0.05, { min: 0.01, max: 0.5, step: 0.01 })
  const toFair = useParam(0.1, { min: 0.01, max: 0.5, step: 0.01 })
  const six = useParam(0.5, { min: 0.2, max: 0.95, step: 0.01 })
  const [rolls, setRolls] = useState(START)

  const r = useMemo(() => {
    const m = casino(toLoaded.value, toFair.value, six.value)
    const xs = rolls.map((f) => f - 1)
    const fb = forwardBackward(m, xs)
    const vit = viterbi(m, xs)
    return { fb, vit }
  }, [toLoaded.value, toFair.value, six.value, rolls])

  const positions = rolls.map((_, n) => n + 1)
  // Heatmap rows are states (0 fair, 1 loaded), columns positions: z[state][n] = p(y_n = state | x).
  const z = STATES.map((v) => r.fb.marginal.map((p) => p[v]))
  const overlay: HeatmapOverlay[] = [
    { name: 'Viterbi path', type: 'line', x: positions, y: r.vit.path, showPoints: true, emphasis: true },
  ]
  const lines: XYSeries[] = [
    { name: 'filtering p(loaded | x₁…xₙ)', type: 'line', x: positions, y: r.fb.filtered.map((p) => p[1]), slot: 1 },
    { name: 'smoothing p(loaded | x)', type: 'line', x: positions, y: r.fb.marginal.map((p) => p[1]), slot: 0 },
  ]
  const cycle = (n: number) => setRolls((rs) => rs.map((f, i) => (i === n ? (f % 6) + 1 : f)))

  return (
    <Interactive
      title="Forward–backward and Viterbi on a short sequence"
      caption="Top: the posterior probability of each die at each roll (dark = probable), with the Viterbi path drawn on top. Middle: filtering uses only the rolls so far (α ⊙ ψ); smoothing also uses the rolls after (α ⊙ ψ ⊙ β), so it can revise an early roll once later rolls arrive. Bottom: the normalised vectors themselves. Click a roll to change its face, and move the sliders to change the model."
      controls={
        <>
          <ParamSlider label="P(fair → loaded)" param={toLoaded} />
          <ParamSlider label="P(loaded → fair)" param={toFair} />
          <ParamSlider label="P(six | loaded)" param={six} />
        </>
      }
      readout={
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
      <Heatmap
        x={positions}
        y={STATES}
        z={z}
        range={[0, 1]}
        xLabel="position n"
        yLabel="die (0 fair, 1 loaded)"
        valueLabel="p(yₙ | x)"
        overlay={overlay}
        height={220}
      />
      <XYChart series={lines} xLabel="position n" yLabel="p(loaded)" yRange={[0, 1]} height={220} />
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
    </Interactive>
  )
}
