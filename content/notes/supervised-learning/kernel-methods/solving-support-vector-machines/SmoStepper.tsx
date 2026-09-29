import { useMemo } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Interactive, ParamSlider, Readout, XYChart, useParam, type XYSeries } from '@/components/viz'
import { BOX, Y_RANGE, exact, slackSeries, svmSeries } from './plot'
import { C0, X, Y, dualObjective, gram, smo, weights } from './solver'

const K = gram(X)
const TRACE = smo(X, Y, C0, 1e-12)
const T = TRACE.steps.length
const SUB = '₀₁₂₃₄₅₆₇₈₉'
const sub = (i: number) => String(i + 1).replace(/\d/g, (d) => SUB[Number(d)])

/** α and b after t steps, and the errors E_t = f(x_t) − y_t they give. */
function state(t: number) {
  const alpha = t === 0 ? X.map(() => 0) : TRACE.steps[t - 1].alpha
  const b = t === 0 ? 0 : TRACE.steps[t - 1].b
  const E = X.map((_, s) => alpha.reduce((f, a, r) => f + a * Y[r] * K[r][s], b) - Y[s])
  return { alpha, b, E, w: weights(X, Y, alpha), objective: dualObjective(K, Y, alpha) }
}

/** The pair's move in the (α_i, α_j) plane, parametrised by α_j along the line y_i α_i + y_j α_j = constant. */
function pairGeometry(t: number) {
  const s = TRACE.steps[t]
  const ai = (aj: number) => s.aiOld + Y[s.i] * Y[s.j] * (s.ajOld - aj)
  return { s, ai }
}

// One window for every step, so the box does not jump: it holds [0, C]² and every unclipped optimum.
const LIM = (() => {
  let lo = 0
  let hi = C0
  for (let t = 0; t < T; t++) {
    const { s, ai } = pairGeometry(t)
    lo = Math.min(lo, s.ajUnclipped, ai(s.ajUnclipped))
    hi = Math.max(hi, s.ajUnclipped, ai(s.ajUnclipped))
  }
  // Pad and round outwards to quarter units so that the ticks are round numbers.
  return [Math.floor((lo - 0.1) * 4) / 4, Math.ceil((hi + 0.1) * 4) / 4] as [number, number]
})()
const LIM_Y: [number | undefined, number | undefined] = LIM

/** SMO on the six-point example, one pair update per step, from α = 0 to the exact optimum. */
export function SmoStepper() {
  const step = useParam(0, { min: 0, max: T, step: 1 })
  const t = step.value
  const cur = useMemo(() => state(t), [t])
  const next = t < T ? pairGeometry(t) : null

  const dataSeries = useMemo(() => {
    const out: XYSeries[] = [...svmSeries(X, Y, cur.w, cur.b), ...slackSeries(X, Y, cur.w, cur.b)]
    if (t < T) {
      const { i, j } = TRACE.steps[t]
      out.push({ name: 'next pair', type: 'scatter', x: [X[i][0], X[j][0]], y: [X[i][1], X[j][1]], emphasis: true })
    }
    return out
  }, [cur, t])

  const boxSeries = useMemo(() => {
    if (t >= T) return null
    const { s, ai } = pairGeometry(t)
    const line = { x: LIM.map(ai), y: [...LIM] }
    const series: XYSeries[] = [
      { name: 'box [0, C]²', type: 'line', x: [0, C0, C0, 0, 0], y: [0, 0, C0, C0, 0], muted: true },
      { name: 'constraint line', type: 'line', ...line, muted: true, dashed: true },
      { name: 'segment [L, H]', type: 'line', x: [ai(s.L), ai(s.H)], y: [s.L, s.H], slot: 2 },
      { name: 'before', type: 'scatter', x: [s.aiOld], y: [s.ajOld], muted: true },
      { name: 'unclipped', type: 'scatter', x: [ai(s.ajUnclipped)], y: [s.ajUnclipped], slot: 3 },
      { name: 'after', type: 'scatter', x: [s.ai], y: [s.aj], emphasis: true },
    ]
    return series
  }, [t])

  const curveSeries = useMemo(() => {
    if (t >= T) return null
    const { s, ai } = pairGeometry(t)
    const alpha = [...cur.alpha]
    const D = (aj: number) => {
      alpha[s.i] = ai(aj)
      alpha[s.j] = aj
      return dualObjective(K, Y, alpha)
    }
    const grid = (a: number, b: number, n: number) => Array.from({ length: n + 1 }, (_, k) => a + ((b - a) * k) / n)
    const lo = Math.min(s.L, s.ajOld, s.ajUnclipped) - 0.25
    const hi = Math.max(s.H, s.ajOld, s.ajUnclipped) + 0.25
    const all = grid(lo, hi, 80)
    const inside = grid(s.L, s.H, 20)
    const series: XYSeries[] = [
      { name: 'D along the line', type: 'line', x: all, y: all.map(D), muted: true },
      { name: 'segment [L, H]', type: 'line', x: inside, y: inside.map(D), slot: 2 },
      { name: 'before', type: 'scatter', x: [s.ajOld], y: [D(s.ajOld)], muted: true },
      { name: 'unclipped', type: 'scatter', x: [s.ajUnclipped], y: [D(s.ajUnclipped)], slot: 3 },
      { name: 'after', type: 'scatter', x: [s.aj], y: [D(s.aj)], emphasis: true },
    ]
    return series
  }, [cur, t])

  const up = (r: number) => (Y[r] > 0 ? cur.alpha[r] < C0 : cur.alpha[r] > 0)
  const low = (r: number) => (Y[r] > 0 ? cur.alpha[r] > 0 : cur.alpha[r] < C0)
  const s = next?.s

  return (
    <Interactive
      title="SMO, one pair at a time"
      caption="Step through SMO from α = 0 with C = 1/2. The left plot shows the boundary after t steps and marks the next pair: i has the smallest error E among the points in I_up, j the largest among those in I_low. The right plot shows the pair in the (αᵢ, αⱼ) plane: the constraint Σ αₜyₜ = 0 keeps it on the dashed line, the box [0, C]² cuts that line to the feasible segment, and the update jumps to the unclipped optimum or, if that lies outside, to the nearest end. The plot below them shows the dual objective D along the same line: a parabola with curvature −η."
      controls={<ParamSlider label="SMO steps taken" param={step} format={(v) => `${v} of ${T}`} withArrows />}
      readout={
        s ? (
          <>
            <Readout label="next pair (i, j)" value={`(x${s.i + 1}, x${s.j + 1})`} />
            <Readout label={`Eᵢ, Eⱼ`} value={`${exact(s.E[s.i])}, ${exact(s.E[s.j])}`} />
            <Readout label="gap Eⱼ − Eᵢ" value={exact(s.gap)} />
            <Readout label="η = ‖xᵢ − xⱼ‖²" value={exact(s.eta)} />
            <Readout label="αⱼ unclipped" value={exact(s.ajUnclipped)} />
            <Readout label="[L, H]" value={`[${exact(s.L)}, ${exact(s.H)}]`} />
            <Readout label="αⱼ, αᵢ after" value={`${exact(s.aj)}, ${exact(s.ai)}`} />
            <Readout label="b after" value={exact(s.b)} />
            <Readout label="dual objective now" value={exact(cur.objective)} />
          </>
        ) : (
          <>
            <Readout label="gap" value="0: every E in I_up ≥ every E in I_low, so KKT holds" />
            <Readout label="dual objective" value={exact(cur.objective)} />
            <Readout label="w" value={`(${exact(cur.w[0])}, ${exact(cur.w[1])})`} />
            <Readout label="b" value={exact(cur.b)} />
          </>
        )
      }
    >
      <div className="grid gap-4 md:grid-cols-2 md:items-start">
        <XYChart
          series={dataSeries}
          xLabel="x₁"
          yLabel="x₂"
          xRange={BOX.x}
          yRange={Y_RANGE}
          equalAspect
          ariaLabel="The data, the current boundary and the next SMO pair"
        />
        {boxSeries && s ? (
          <XYChart
            series={boxSeries}
            xLabel={`α${sub(s.i)}`}
            yLabel={`α${sub(s.j)}`}
            xRange={LIM}
            yRange={LIM_Y}
            equalAspect
            ariaLabel="The pair's feasible segment in the box"
          />
        ) : (
          <p className="self-center text-center text-xs text-muted-foreground">
            SMO has stopped. No pair can raise the dual objective.
          </p>
        )}
      </div>
      {curveSeries && s ? (
        <XYChart
          series={curveSeries}
          xLabel={`α${sub(s.j)}`}
          yLabel="D"
          height={240}
          ariaLabel="The dual objective along the pair's line"
        />
      ) : null}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>point</TableHead>
            <TableHead className="text-right">y</TableHead>
            <TableHead className="text-right">α</TableHead>
            <TableHead className="text-right">E = f − y</TableHead>
            <TableHead className="text-center">in I_up</TableHead>
            <TableHead className="text-center">in I_low</TableHead>
            <TableHead className="text-right">ξ = max(0, −yE)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {X.map((p, r) => (
            <TableRow key={r} className={s && (r === s.i || r === s.j) ? 'bg-muted' : undefined}>
              <TableCell className="py-1 font-mono text-xs">
                x{r + 1} ({p[0]}, {p[1]})
              </TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{Y[r] > 0 ? '+1' : '−1'}</TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{exact(cur.alpha[r])}</TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{exact(cur.E[r])}</TableCell>
              <TableCell className="py-1 text-center text-xs">{up(r) ? '✓' : ''}</TableCell>
              <TableCell className="py-1 text-center text-xs">{low(r) ? '✓' : ''}</TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">
                {exact(Math.max(0, -Y[r] * cur.E[r]))}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Interactive>
  )
}
