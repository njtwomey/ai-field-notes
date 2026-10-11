import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

type Fn = { label: string; f: (x: number) => number; df: (x: number) => number }

const FUNCTIONS = {
  exp: { label: 'eˣ at a = 1', f: Math.exp, df: Math.exp },
  sin: { label: 'sin x at a = 1', f: Math.sin, df: Math.cos },
  sqrt: { label: '√x at a = 1', f: Math.sqrt, df: (x) => 0.5 / Math.sqrt(x) },
} satisfies Record<string, Fn>

type Name = keyof typeof FUNCTIONS

const A = 1
const LOG_H = Array.from({ length: 321 }, (_, i) => -16 + i * 0.05)
const HS = LOG_H.map((l) => 10 ** l)
/** Errors below this are drawn at it, so an exact hit does not fall off the log axis. */
const FLOOR = 1e-14

const SUPERSCRIPT: Record<string, string> = {
  '-': '⁻',
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '.': '·',
}

/** Tick label for log₁₀ h = v: the power of ten it stands for. */
const powerOfTen = (v: number) => {
  const r = Math.round(v * 10) / 10
  return r === 0 ? '1' : `10${[...String(r)].map((c) => SUPERSCRIPT[c] ?? c).join('')}`
}

/** Every arithmetic step rounded to the chosen precision: float32 via Math.fround, float64 as JavaScript computes. */
const rounder = (single: boolean) => (single ? Math.fround : (v: number) => v)

function differences(fn: Fn, h: number, single: boolean) {
  const r = rounder(single)
  const f = (x: number) => r(fn.f(r(x)))
  const a = r(A)
  const step = r(h)
  const forward = r(r(f(r(a + step)) - f(a)) / step)
  const central = r(r(f(r(a + step)) - f(r(a - step))) / r(2 * step))
  return { forward, central }
}

/**
 * The error of the forward and central difference quotients against h, on log–log axes. Truncation error falls as h
 * shrinks (slope 1 for forward, 2 for central); rounding error grows like ε/h; the sum is a V with its floor near
 * h ≈ √ε (forward) or ε^(1/3) (central).
 */
export function FiniteDifferenceError() {
  const state = useFigureState({
    name: choice<Name>(
      (Object.keys(FUNCTIONS) as Name[]).map((value) => ({ value, label: FUNCTIONS[value].label })),
      'exp',
      { label: 'function' },
    ),
    single: setting(false, 'single precision (float32)'),
    logH: slider(-16, 0, -5, { step: 0.05, label: 'log₁₀ h', onChart: true }),
  })
  const fn: Fn = FUNCTIONS[state.name]
  const exact = fn.df(A)

  const errors = useMemo(() => {
    const fwd: number[] = []
    const ctr: number[] = []
    for (const h of HS) {
      const d = differences(fn, h, state.single)
      fwd.push(Math.max(FLOOR, Math.abs(d.forward - exact)))
      ctr.push(Math.max(FLOOR, Math.abs(d.central - exact)))
    }
    const best = (e: number[]) => HS[e.indexOf(Math.min(...e))]
    return { fwd, ctr, bestFwd: best(fwd), bestCtr: best(ctr) }
  }, [fn, state.single, exact])

  const h = 10 ** state.logH
  const now = differences(fn, h, state.single)

  const x = useAxis({ label: 'step h (log scale)', range: [-16, 0], format: powerOfTen })
  const y = useAxis({ label: 'absolute error', range: [1e-14, 100], log: true })

  return (
    <Figure
      title="Choosing the step of a finite difference"
      state={state}
      caption="The error of the forward quotient (f(a + h) − f(a)) / h and the central quotient (f(a + h) − f(a − h)) / 2h against the exact derivative, on log–log axes. Read from the right: for large h the truncation error falls with slope 1 (forward) and 2 (central). Below the best h, rounding takes over: f(a + h) and f(a) agree in most of their digits, the subtraction leaves only noise, and dividing by a tiny h magnifies it, so the error grows like 1/h. At the far left, h is below the spacing of floating-point numbers near a, so a + h rounds to a and the step is lost entirely. Drag the vertical line to choose h. Single precision moves the best steps to much larger h."
      readouts={
        <>
          <Readout label="h" value={formatNumber(h)} />
          <Readout label="exact f'(a)" value={exact.toPrecision(12)} />
          <Readout label="forward" value={now.forward.toPrecision(12)} />
          <Readout label="central" value={now.central.toPrecision(12)} />
          <Readout
            label="best h (forward, central)"
            value={`${formatNumber(errors.bestFwd)}, ${formatNumber(errors.bestCtr)}`}
          />
        </>
      }
    >
      <Plot x={x} y={y} height={340}>
        <Curve name="forward difference error" x={LOG_H} y={errors.fwd} slot={0} />
        <Curve name="central difference error" x={LOG_H} y={errors.ctr} slot={1} />
        <Handle {...state.handle('logH', { label: 'h' })} />
      </Plot>
    </Figure>
  )
}
