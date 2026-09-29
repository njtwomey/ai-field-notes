import { useMemo, useState } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Interactive, ParamButton, ParamChoice, Readout, XYChart } from '@/components/viz'
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

/** Guess each point's status, solve the linear KKT equations that the guess implies, and check the rest. */
export function ActiveSetGuess() {
  const [status, setStatus] = useState<Status[]>(START)
  const r = useMemo(() => activeSet(X, Y, C0, status), [status])
  const series = useMemo(
    () => (r.ok ? [...svmSeries(X, Y, r.w, r.b), ...slackSeries(X, Y, r.w, r.b)] : svmSeries(X, Y, [0, 0], 0)),
    [r],
  )
  const failed = r.ok ? r.checks.filter((c) => !c.holds).length : null

  return (
    <Interactive
      title="Guess the active set, then check it"
      caption="Each point is guessed to be at α = 0, free (on the margin) or at α = C, with C = 1/2. The guess turns the KKT conditions into linear equations: yᵢf(xᵢ) = 1 for each free point and Σ αᵢyᵢ = 0. Their solution gives α, w and b. The guess is right only if the conditions it did not impose also hold: y f ≥ 1 where α = 0, y f ≤ 1 where α = C, and 0 ≤ α ≤ C for the free points. The figure starts from the wrong guess worked in the text."
      controls={
        <>
          {X.map((_, t) => (
            <ParamChoice
              key={t}
              label={`x${t + 1} (y = ${Y[t] > 0 ? '+1' : '−1'})`}
              value={status[t]}
              onChange={(v) => setStatus((prev) => prev.map((s, k) => (k === t ? v : s)))}
              options={OPTIONS}
            />
          ))}
          <ParamButton onClick={() => setStatus(START)}>Wrong guess</ParamButton>
          <ParamButton onClick={() => setStatus(RIGHT)}>Right guess</ParamButton>
        </>
      }
      readout={
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
          <XYChart
            series={series}
            xLabel="x₁"
            yLabel="x₂"
            xRange={BOX.x}
            yRange={Y_RANGE}
            equalAspect
            ariaLabel="The boundary implied by the guessed active set"
          />
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
    </Interactive>
  )
}
