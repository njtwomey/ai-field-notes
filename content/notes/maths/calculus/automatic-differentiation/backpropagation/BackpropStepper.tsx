import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, StepControls, formatNumber } from 'aifn-render'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { cn } from '@/lib/utils'

type Node = { name: string; expr: string; value: number; local: string; adjoint: number; adjointExpr: string }

/** f(x, y) = (x·y + sin x)², as a graph of primitive operations with forward values and reverse-mode adjoints. */
function trace(x: number, y: number): Node[] {
  const a = x * y
  const b = Math.sin(x)
  const c = a + b
  const f = c * c
  // Adjoints ∂f/∂node, computed from the output backwards.
  const fBar = 1
  const cBar = 2 * c * fBar
  const aBar = cBar
  const bBar = cBar
  const xBar = aBar * y + bBar * Math.cos(x)
  const yBar = aBar * x
  return [
    { name: 'x', expr: 'input', value: x, local: '—', adjoint: xBar, adjointExpr: 'ā·y + b̄·cos x' },
    { name: 'y', expr: 'input', value: y, local: '—', adjoint: yBar, adjointExpr: 'ā·x' },
    { name: 'a', expr: 'x · y', value: a, local: '∂a/∂x = y, ∂a/∂y = x', adjoint: aBar, adjointExpr: 'c̄ · 1' },
    { name: 'b', expr: 'sin x', value: b, local: '∂b/∂x = cos x', adjoint: bBar, adjointExpr: 'c̄ · 1' },
    { name: 'c', expr: 'a + b', value: c, local: '∂c/∂a = 1, ∂c/∂b = 1', adjoint: cBar, adjointExpr: 'f̄ · 2c' },
    { name: 'f', expr: 'c²', value: f, local: '∂f/∂c = 2c', adjoint: fBar, adjointExpr: '1 (seed)' },
  ]
}

/** Forward pass visits nodes 0→5; the backward pass visits them 5→0. */
const FORWARD = [0, 1, 2, 3, 4, 5]
const BACKWARD = [5, 4, 3, 2, 1, 0]
const STEPS = FORWARD.length + BACKWARD.length

export function BackpropStepper() {
  const [x, setX] = useState(1.5)
  const [y, setY] = useState(-0.5)
  const [step, setStep] = useState(0)
  const nodes = useMemo(() => trace(x, y), [x, y])

  const valueShown = new Set(FORWARD.slice(0, Math.min(step, FORWARD.length)))
  const adjointShown = new Set(BACKWARD.slice(0, Math.max(0, step - FORWARD.length)))
  const current = step === 0 ? -1 : step <= FORWARD.length ? FORWARD[step - 1] : BACKWARD[step - FORWARD.length - 1]
  const phase = step <= FORWARD.length ? 'forward' : 'backward'

  const f = (u: number, v: number) => (u * v + Math.sin(u)) ** 2
  const eps = 1e-5
  const numeric = [(f(x + eps, y) - f(x - eps, y)) / (2 * eps), (f(x, y + eps) - f(x, y - eps)) / (2 * eps)]

  return (
    <Interactive
      title="Backpropagation, one node at a time"
      caption="For f(x, y) = (xy + sin x)², the forward pass fills in each node's value from the inputs. The backward pass then fills in each node's adjoint, ∂f/∂node, starting from f̄ = 1 and multiplying by one local derivative per edge. The adjoints of x and y are the gradient."
      controls={
        <>
          <ParamSlider label="input x" value={x} onChange={setX} min={-3} max={3} step={0.1} />
          <ParamSlider label="input y" value={y} onChange={setY} min={-3} max={3} step={0.1} />
          <StepControls
            onStep={() => setStep((s) => Math.min(s + 1, STEPS))}
            onRun={() => setStep(STEPS)}
            onReset={() => setStep(0)}
            done={step === STEPS}
          />
        </>
      }
      readout={
        <>
          <Readout label="step" value={`${step} of ${STEPS} (${phase})`} />
          <Readout label="∂f/∂x, finite difference" value={formatNumber(numeric[0])} />
          <Readout label="∂f/∂y, finite difference" value={formatNumber(numeric[1])} />
        </>
      }
    >
      <div className="overflow-x-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>node</TableHead>
              <TableHead>computes</TableHead>
              <TableHead>value</TableHead>
              <TableHead>local derivatives</TableHead>
              <TableHead>adjoint rule</TableHead>
              <TableHead>adjoint ∂f/∂node</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {nodes.map((n, i) => (
              <TableRow key={n.name} className={cn(i === current && 'bg-muted')}>
                <TableCell className="font-mono">{n.name}</TableCell>
                <TableCell className="font-mono text-xs">{n.expr}</TableCell>
                <TableCell className="font-mono tabular-nums">
                  {valueShown.has(i) ? formatNumber(n.value) : '·'}
                </TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground">{n.local}</TableCell>
                <TableCell className="font-mono text-xs text-muted-foreground">{n.adjointExpr}</TableCell>
                <TableCell className="font-mono tabular-nums">
                  {adjointShown.has(i) ? formatNumber(n.adjoint) : '·'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Interactive>
  )
}
