import { useMemo } from 'react'
import {
  Button,
  choice,
  Figure,
  Plot,
  Readout,
  seriesLayers,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { BOX, Y_RANGE, exact, slackSeries, svmSeries } from './plot'
import { C0, X, Y, activeSet, type Status } from './solver'

const OPTIONS = [
  { value: 'zero', label: 'α = 0' },
  { value: 'free', label: 'free' },
  { value: 'bound', label: 'α = C' },
] as const satisfies readonly { value: Status; label: string }[]

/** The wrong guess worked in the text: x₂ on the margin rather than inside it. */
const START: Status[] = ['zero', 'free', 'free', 'bound', 'zero', 'free']
const RIGHT: Status[] = ['zero', 'bound', 'free', 'bound', 'zero', 'free']

/** The guess for point t as a choice field, labelled with the point's class. */
const guess = (t: number) => choice<Status>(OPTIONS, START[t], { label: `x${t + 1} (y = ${Y[t] > 0 ? '+1' : '−1'})` })

/** Guess each point's status, solve the linear KKT equations that the guess implies, and check the rest. */
export function ActiveSetGuess() {
  const state = useFigureState({ x1: guess(0), x2: guess(1), x3: guess(2), x4: guess(3), x5: guess(4), x6: guess(5) })
  const { x1, x2, x3, x4, x5, x6 } = state
  const status: Status[] = useMemo(() => [x1, x2, x3, x4, x5, x6], [x1, x2, x3, x4, x5, x6])
  const setGuess = (guesses: Status[]) =>
    guesses.forEach((g, t) => state.set(`x${t + 1}` as 'x1' | 'x2' | 'x3' | 'x4' | 'x5' | 'x6', g))
  const r = useMemo(() => activeSet(X, Y, C0, status), [status])
  const series = useMemo(
    () => (r.ok ? [...svmSeries(X, Y, r.w, r.b), ...slackSeries(X, Y, r.w, r.b)] : svmSeries(X, Y, [0, 0], 0)),
    [r],
  )
  const failed = r.ok ? r.checks.filter((c) => !c.holds).length : null

  const xAxis = useAxis({ label: 'x₁', range: BOX.x })
  const yAxis = useAxis({ label: 'x₂', range: Y_RANGE, equal: xAxis })
  return (
    <Figure
      title="Guess the active set, then check it"
      caption="Each point is guessed to be at α = 0, free (on the margin) or at α = C, with C = 1/2. The guess turns the KKT conditions into linear equations: yᵢf(xᵢ) = 1 for each free point and Σ αᵢyᵢ = 0. Their solution gives α, w and b. The guess is right only if the conditions it did not impose also hold: y f ≥ 1 where α = 0, y f ≤ 1 where α = C, and 0 ≤ α ≤ C for the free points. The figure starts from the wrong guess worked in the text."
      state={state}
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => setGuess(START)}>
            Wrong guess
          </Button>
          <Button variant="outline" size="sm" onClick={() => setGuess(RIGHT)}>
            Right guess
          </Button>
        </>
      }
      readouts={
        r.ok ? (
          <>
            <Readout label="w" value={`(${exact(r.w[0])}, ${exact(r.w[1])})`} />
            <Readout label="b" value={exact(r.b)} />
            <Readout
              label="verdict"
              value={failed === 0 ? 'every KKT condition holds: this is the solution' : `${failed} condition(s) fail`}
            />
          </>
        ) : (
          <Readout label="verdict" value={r.reason} />
        )
      }
    >
      <div className="flex flex-col gap-4">
        <div className="mx-auto w-full max-w-lg">
          <Plot x={xAxis} y={yAxis} ariaLabel={'The boundary implied by the guessed active set'}>
            {seriesLayers(series)}
          </Plot>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>point</TableHead>
              <TableHead>guess</TableHead>
              <TableHead className="text-right">α</TableHead>
              <TableHead className="text-right">y f(x)</TableHead>
              <TableHead>check</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {X.map((_, t) => (
              <TableRow key={t}>
                <TableCell className="py-1 font-mono text-xs">x{t + 1}</TableCell>
                <TableCell className="py-1 text-xs">{OPTIONS.find((o) => o.value === status[t])!.label}</TableCell>
                <TableCell className="py-1 text-right font-mono text-xs tabular-nums">
                  {r.ok ? exact(r.alpha[t]) : '–'}
                </TableCell>
                <TableCell className="py-1 text-right font-mono text-xs tabular-nums">
                  {r.ok ? exact(r.margin[t]) : '–'}
                </TableCell>
                <TableCell className={r.ok && !r.checks[t].holds ? 'py-1 text-xs text-destructive' : 'py-1 text-xs'}>
                  {r.ok ? `${r.checks[t].requirement} ${r.checks[t].holds ? '✓' : '✗'}` : ''}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Figure>
  )
}
