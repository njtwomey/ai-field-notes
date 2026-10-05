import type { LinearProgramResult, QuadraticProgramResult } from 'aifn-compute/optim/programming'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { ReactNode } from 'react'
import { Badge } from 'aifn-render/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from 'aifn-render/ui/table'
import { formatValue } from './format'

export type ProgramViewProps = {
  /** A `linprog` or `quadprog` result. */
  result: LinearProgramResult | QuadraticProgramResult
  /** Names of the variables (default x1, x2, …). */
  variableNames?: readonly string[]
  /** Names of the inequality rows (default row 1, row 2, …). */
  constraintNames?: readonly string[]
  /** Names of the equality rows (default eq 1, eq 2, …). */
  equalityNames?: readonly string[]
}

const flat = (t: Tensor | null | undefined) => (t ? toFlat(t) : [])

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="min-w-0 space-y-1">
      <div className="text-xs font-medium text-muted-foreground">{title}</div>
      <div className="overflow-x-auto">{children}</div>
    </div>
  )
}

function Grid({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <Table className="text-xs">
      <TableHeader>
        <TableRow>
          {head.map((h, j) => (
            // Headers can repeat (e.g. two unnamed columns), so the key is the position.
            <TableHead key={j} className="h-7 px-2 text-xs">
              {h}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((r, i) => (
          <TableRow key={i}>
            {r.map((c, j) => (
              <TableCell key={j} className={j === 0 ? 'px-2 py-1' : 'px-2 py-1 text-right font-mono tabular-nums'}>
                {c}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function Check({ label, value }: { label: string; value: number }) {
  const ok = Math.abs(value) < 1e-6
  return (
    <span className="flex items-center gap-1.5 text-xs">
      <span className={ok ? 'text-muted-foreground' : 'font-medium'}>{label}</span>
      <span className="font-mono tabular-nums">{formatValue(value)}</span>
    </span>
  )
}

const isQuadratic = (r: LinearProgramResult | QuadraticProgramResult): r is QuadraticProgramResult =>
  r.report !== null && 'lambda' in r.report

const active = (on: number) => (on ? <Badge variant="secondary">active</Badge> : null)

/**
 * The solution of a linear or quadratic program: status, objective, the variables (with reduced costs and bound duals
 * for an LP), each inequality's slack, dual and whether it is active, the equalities' duals, and the optimality
 * checks (duality gap and complementary slackness for an LP; KKT residuals for a QP).
 */
export function ProgramView({ result, variableNames, constraintNames, equalityNames }: ProgramViewProps) {
  const x = flat(result.x)
  const name = (i: number) => variableNames?.[i] ?? `x${i + 1}`
  const rowName = (i: number) => constraintNames?.[i] ?? `row ${i + 1}`
  const eqName = (i: number) => equalityNames?.[i] ?? `eq ${i + 1}`
  const header = (
    <div className="flex flex-wrap items-center gap-3 text-sm">
      <Badge variant={result.status === 'optimal' ? 'default' : 'destructive'}>{result.status}</Badge>
      <span>
        objective <span className="font-mono tabular-nums">{formatValue(result.objective)}</span>
      </span>
      <span className="text-xs text-muted-foreground">
        {result.method}, {result.steps} steps
      </span>
    </div>
  )

  if (isQuadratic(result)) {
    // Quadratic program: KKT report.
    const r = result.report
    const slack = flat(r.slack)
    const lambda = flat(r.lambda)
    const on = flat(r.active)
    const nu = flat(r.nu)
    return (
      <div className="space-y-3">
        {header}
        <div className="grid gap-4 md:grid-cols-2">
          <Section title="variables">
            <Grid head={['', 'value']} rows={x.map((v, i) => [name(i), formatValue(v)])} />
          </Section>
          {slack.length > 0 && (
            <Section title="inequalities Ax ≤ b">
              <Grid
                head={['', 'slack', 'λ', '']}
                rows={slack.map((s, i) => [rowName(i), formatValue(s), formatValue(lambda[i]), active(on[i])])}
              />
            </Section>
          )}
          {nu.length > 0 && (
            <Section title="equalities Ex = e">
              <Grid head={['', 'ν']} rows={nu.map((v, i) => [eqName(i), formatValue(v)])} />
            </Section>
          )}
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1">
          <Check label="stationarity" value={r.stationarity} />
          <Check label="primal infeasibility" value={r.primalInfeasibility} />
          <Check label="dual infeasibility" value={r.dualInfeasibility} />
          <Check label="complementarity" value={r.complementarity} />
        </div>
      </div>
    )
  }

  const r = result.report
  if (!r)
    return (
      <div className="space-y-2">
        {header}
        {result.ray && (
          <div className="text-xs text-muted-foreground">
            unbounded along d = ({flat(result.ray).map(formatValue).join(', ')})
          </div>
        )}
      </div>
    )
  const reduced = flat(r.reducedCosts)
  const lower = flat(r.duals.lower)
  const upper = flat(r.duals.upper)
  const slack = flat(r.slack)
  const ineq = flat(r.duals.ineq)
  const on = flat(r.active)
  const eq = flat(r.duals.eq)
  return (
    <div className="space-y-3">
      {header}
      <div className="grid gap-4 md:grid-cols-2">
        <Section title="variables">
          <Grid
            head={['', 'value', 'reduced cost', 'lower dual', 'upper dual']}
            rows={x.map((v, i) => [
              name(i),
              formatValue(v),
              formatValue(reduced[i]),
              formatValue(lower[i]),
              formatValue(upper[i]),
            ])}
          />
        </Section>
        {slack.length > 0 && (
          <Section title="inequalities A_ub x ≤ b_ub">
            <Grid
              head={['', 'slack', 'dual', '']}
              rows={slack.map((s, i) => [rowName(i), formatValue(s), formatValue(ineq[i]), active(on[i])])}
            />
          </Section>
        )}
        {eq.length > 0 && (
          <Section title="equalities A_eq x = b_eq">
            <Grid
              head={['', 'residual', 'dual']}
              rows={eq.map((v, i) => [eqName(i), formatValue(flat(r.residualEq)[i]), formatValue(v)])}
            />
          </Section>
        )}
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1">
        <span className="text-xs">
          dual objective <span className="font-mono tabular-nums">{formatValue(r.dualObjective)}</span>
        </span>
        <Check label="duality gap" value={r.dualityGap} />
        <Check label="primal infeasibility" value={r.primalInfeasibility} />
        <Check label="dual infeasibility" value={r.dualInfeasibility} />
        <Check label="complementarity" value={r.complementarity} />
      </div>
    </div>
  )
}
