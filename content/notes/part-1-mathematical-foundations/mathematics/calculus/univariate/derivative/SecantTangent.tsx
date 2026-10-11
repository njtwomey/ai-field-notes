import { useContext, useMemo, type ReactNode } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  Plot,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

type Fn = {
  label: string
  f: (x: number) => number
  /** The derivative, or undefined where it does not exist. */
  df: (x: number) => number | undefined
  /** Why the derivative fails to exist where `df` is undefined. */
  why?: string
  /** True where the tangent exists but is vertical (infinite slope). */
  vertical?: (x: number) => boolean
  range: [number, number]
}

const KINK = 1e-9

const FUNCTIONS = {
  square: { label: 'x²', f: (x) => x * x, df: (x) => 2 * x, range: [-1, 9] },
  cube: { label: 'x³', f: (x) => x * x * x, df: (x) => 3 * x * x, range: [-9, 9] },
  sin: { label: 'sin x', f: Math.sin, df: Math.cos, range: [-1.6, 1.6] },
  exp: { label: 'eˣ', f: Math.exp, df: Math.exp, range: [-1, 10] },
  abs: {
    label: '|x|',
    f: Math.abs,
    df: (x) => (Math.abs(x) < KINK ? undefined : Math.sign(x)),
    why: 'no limit: −1 from the left, +1 from the right',
    range: [-0.5, 3],
  },
  cbrt: {
    label: '∛x',
    f: Math.cbrt,
    df: (x) => (Math.abs(x) < KINK ? undefined : 1 / (3 * Math.cbrt(x) ** 2)),
    why: 'the quotient grows without bound: a vertical tangent',
    vertical: (x) => Math.abs(x) < KINK,
    range: [-1.8, 1.8],
  },
} satisfies Record<string, Fn>

type Name = keyof typeof FUNCTIONS

const DOMAIN: [number, number] = [-3, 3]
const H_MAX = 1
const PANEL = 320
const XS = Array.from({ length: 301 }, (_, i) => DOMAIN[0] + ((DOMAIN[1] - DOMAIN[0]) * i) / 300)
/** Steps for the quotient panel; h = 0 is left out (the quotient is 0/0 there), which breaks the curve. */
const HS = Array.from({ length: 401 }, (_, i) => -H_MAX + (2 * H_MAX * i) / 400)

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * The secant through P = (a, f(a)) and Q = (a + h, f(a + h)) turning into the tangent as h shrinks. The right panel
 * plots the secant's slope against h: the derivative is the height that curve approaches at h = 0.
 */
export function SecantTangent() {
  const state = useFigureState({
    name: choice<Name>(
      (Object.keys(FUNCTIONS) as Name[]).map((value) => ({ value, label: FUNCTIONS[value].label })),
      'square',
      { label: 'function f(x)' },
    ),
    a: slider(-2, 2, 1, { step: 0.05, label: 'point a', onChart: true }),
    h: slider(-H_MAX, H_MAX, 1, { step: 0.01, label: 'step h', onChart: true }),
  })
  const fn: Fn = FUNCTIONS[state.name]
  const { a, h } = state
  const setA = state.bind('a').set
  const setH = state.bind('h').set

  const fa = fn.f(a)
  const fq = fn.f(a + h)
  const zeroStep = Math.abs(h) < 1e-9
  const slope = zeroStep ? undefined : (fq - fa) / h
  const derivative = fn.df(a)
  const vertical = fn.vertical?.(a) ?? false

  const curve = useMemo(() => XS.map(fn.f), [fn])
  const line = (m: number) => DOMAIN.map((x) => fa + m * (x - a))

  // The quotient as a function of h at this a, with a gap at h = 0 where it is undefined.
  const quotient = useMemo(() => HS.map((s) => (Math.abs(s) < 1e-9 ? NaN : (fn.f(a + s) - fn.f(a)) / s)), [fn, a])
  const qRange = useMemo((): [number, number] => {
    const finite = quotient.filter((q, i) => Number.isFinite(q) && Math.abs(HS[i]) >= 0.1)
    let lo = Math.min(...finite)
    let hi = Math.max(...finite)
    const d = fn.df(a)
    if (d !== undefined) {
      lo = Math.min(lo, d)
      hi = Math.max(hi, d)
    }
    const pad = Math.max(0.5, 0.15 * (hi - lo))
    return [lo - pad, hi + pad]
  }, [quotient, fn, a])

  const x = useAxis({ label: 'x', range: DOMAIN })
  const y = useAxis({ label: 'y = f(x)', range: fn.range })
  const hx = useAxis({ label: 'step h', range: [-H_MAX, H_MAX] })
  const qy = useAxis({ label: 'secant slope', range: qRange })

  const verdict = vertical
    ? 'does not exist (vertical tangent)'
    : derivative === undefined
      ? 'does not exist'
      : formatNumber(derivative)

  return (
    <Figure
      title="From secant to tangent"
      state={state}
      caption={
        <>
          Left: the secant (solid) is the straight line through P = (a, f(a)) and Q = (a + h, f(a + h)). Its slope is
          rise over run, (f(a + h) − f(a)) / h; the dotted legs show the run h and the rise. Drag P along the curve to
          move a, and drag Q towards P to shrink h. As Q closes in on P from either side, the secant turns into the
          tangent (dashed), whose slope is f'(a). Right: the secant slope for every h at this a. It is undefined at h =
          0 (a gap), and f'(a) is the height the curve approaches there. For |x| at a = 0 the two sides approach −1 and
          +1, so there is no limit; for ∛x at a = 0 both sides shoot up and the tangent is vertical.
        </>
      }
      readouts={
        <>
          <Readout
            label="secant slope"
            value={slope === undefined ? '0/0 (undefined at h = 0)' : formatNumber(slope)}
          />
          <Readout label="f'(a)" value={verdict} />
          <Readout
            label="gap"
            value={slope === undefined || derivative === undefined ? '—' : formatNumber(Math.abs(slope - derivative))}
          />
          {derivative === undefined && fn.why && <Readout label="why" value={fn.why} />}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={x} y={y} title="secant and tangent">
            <Curve name={`f(x) = ${fn.label}`} x={XS} y={curve} emphasis />
            {slope !== undefined && <Curve name="secant" x={DOMAIN} y={line(slope)} slot={0} live />}
            {derivative !== undefined && <Curve name="tangent" x={DOMAIN} y={line(derivative)} slot={1} dashed live />}
            {vertical && (
              <Segments
                name="tangent"
                segments={[{ from: [a, fn.range[0]], to: [a, fn.range[1]] }]}
                slot={1}
                dashed
                width={2}
                live
              />
            )}
            {!zeroStep && (
              <Segments
                name="run and rise"
                segments={[
                  { from: [a, fa], to: [a + h, fa] },
                  { from: [a + h, fa], to: [a + h, fq] },
                ]}
                dashed
                live
              />
            )}
            <Handle kind="point" at={[a + h, fq]} label="Q" onDrag={([px]) => setH(clamp(px - a, -H_MAX, H_MAX))} />
            <Handle kind="point" at={[a, fa]} label="P" onDrag={([px]) => setA(clamp(px, -2, 2))} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={hx} y={qy} title="secant slope against h">
            <Curve name="secant slope (f(a + h) − f(a)) / h" x={HS} y={quotient} slot={0} />
            {derivative !== undefined && <Annotation y={derivative} text="f'(a)" slot={1} dashed live />}
            <Handle {...state.handle('h', { label: 'h' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
