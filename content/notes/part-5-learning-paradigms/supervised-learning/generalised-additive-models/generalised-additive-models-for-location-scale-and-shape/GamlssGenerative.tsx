import { useMemo, useState } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { fromData, linspace, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { bsplineBasis, clampedKnots } from 'aifn-compute/numerics/interpolate'
import {
  distributionalFamily,
  link,
  type DistributionalFamilyName,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'

const GRID = toFlat(linspace(0, 1, 121))
const DEGREE = 3
const KNOTS = clampedKnots(linspace(0, 1, 4), DEGREE)
/** Six cubic B-splines on [0, 1]; each coefficient is a control point at its Greville abscissa. */
const BASIS = toRows(bsplineBasis(fromData(Float64Array.from(GRID)), KNOTS, DEGREE))
const T = toFlat(KNOTS)
const CONTROL_X = Array.from({ length: BASIS[0].length }, (_, j) => (T[j + 1] + T[j + 2] + T[j + 3]) / 3)
const SYMBOL = { mu: 'μ', sigma: 'σ', nu: 'ν' } as const
const CENTILES = [0.03, 0.1, 0.25, 0.75, 0.9, 0.97]

/** Each family's parameter curves in response space, from which the starting control points are taken. */
const DEFAULTS: Record<DistributionalFamilyName, ((x: number) => number)[]> = {
  normal: [(x) => 2 + 1.5 * Math.sin(1.5 * Math.PI * x), (x) => 0.3 + 0.9 * x * x],
  'student-t': [(x) => 2 + 1.5 * Math.sin(1.5 * Math.PI * x), (x) => 0.3 + 0.9 * x * x, () => 3],
  'box-cox-cole-green': [(x) => 5 + 10 * x, (x) => 0.1 + 0.15 * x, (x) => 1 - 2 * x],
  gamma: [(x) => 2 + 4 * x, (x) => 0.3 + 0.4 * x],
  poisson: [(x) => 2 + 10 * x * x],
}

const linkChoice = (names: LinkName[], label: string) => choice(names, names[0], { label })

export function GamlssGenerative() {
  const state = useFigureState({
    model: variants(
      {
        normal: {
          label: 'Normal',
          params: { mu: linkChoice(['identity', 'log'], 'μ link'), sigma: linkChoice(['log', 'identity'], 'σ link') },
        },
        'student-t': {
          label: 'Student t',
          params: {
            mu: linkChoice(['identity', 'log'], 'μ link'),
            sigma: linkChoice(['log', 'identity'], 'σ link'),
            nu: linkChoice(['log', 'identity'], 'ν link'),
          },
        },
        'box-cox-cole-green': {
          label: 'BCCG (LMS)',
          params: {
            mu: linkChoice(['identity', 'log'], 'μ link'),
            sigma: linkChoice(['log', 'identity'], 'σ link'),
            nu: linkChoice(['identity'], 'ν link'),
          },
        },
        gamma: {
          label: 'Gamma',
          params: { mu: linkChoice(['log', 'identity'], 'μ link'), sigma: linkChoice(['log', 'identity'], 'σ link') },
        },
        poisson: { label: 'Poisson', params: { mu: linkChoice(['log', 'identity', 'sqrt'], 'μ link') } },
      },
      { label: 'family', choiceLabel: 'family' },
    ),
    centiles: setting(false, 'centile curves'),
    density: setting(false, 'density'),
  })

  const name = state.model.key as DistributionalFamilyName
  const family = useMemo(() => distributionalFamily(name), [name])
  const values = state.model.values as Partial<Record<'mu' | 'sigma' | 'nu', LinkName>>
  const linkNames = family.parameters.map((p) => values[p.name] ?? p.links[0])
  const key = `${name}|${linkNames.join(',')}`
  const links = useMemo(
    () =>
      key
        .split('|')[1]
        .split(',')
        .map((l) => link(l as LinkName)),
    [key],
  )

  const [edits, setEdits] = useState<Record<string, number[][]>>({})
  const control = useMemo(
    () =>
      edits[key] ??
      DEFAULTS[name].map((f, k) => CONTROL_X.map((x) => links[k].link(f(Math.min(Math.max(x, 0.01), 0.99))) as number)),
    [edits, key, name, links],
  )
  const move = (k: number, j: number, y: number) =>
    setEdits((e) => ({
      ...e,
      [key]: control.map((c, kk) => (kk === k ? c.map((v, jj) => (jj === j ? Math.min(20, Math.max(-20, y)) : v)) : c)),
    }))

  const curves = useMemo(() => {
    const eta = control.map((c) => BASIS.map((row) => row.reduce((s, b, j) => s + b * c[j], 0)))
    const theta = eta.map((e, k) => e.map((v) => links[k].inverse(v) as number))
    const at = (i: number) => theta.map((t) => t[i])
    const valid = GRID.map((_, i) => family.valid(at(i)))
    const nan = (i: number, v: number) => (valid[i] ? v : NaN)
    const mean = GRID.map((_, i) => nan(i, family.mean(at(i))))
    const sd = GRID.map((_, i) => nan(i, Math.sqrt(family.variance(at(i)))))
    const median = GRID.map((_, i) => nan(i, family.quantile(0.5, at(i))))
    const centiles = CENTILES.map((p) => GRID.map((_, i) => nan(i, family.quantile(p, at(i)))))
    return { eta, theta, valid, mean, sd, median, centiles, at }
  }, [control, links, family])

  const yLimits = useMemo(() => {
    const lo = Math.min(...curves.centiles[0].filter(Number.isFinite))
    const hi = Math.max(...curves.centiles[5].filter(Number.isFinite))
    const pad = 0.25 * (hi - lo || 1)
    return [lo - pad, hi + pad] as const
  }, [curves])

  const density = useMemo(() => {
    if (!state.density) return null
    const xs = GRID.filter((_, i) => i % 2 === 0)
    const discrete = family.support === 'non-negative-integers'
    const ys = discrete
      ? Array.from({ length: Math.max(2, Math.ceil(yLimits[1]) + 1) }, (_, i) => i)
      : Array.from({ length: 100 }, (_, i) => yLimits[0] + ((i + 0.5) * (yLimits[1] - yLimits[0])) / 100)
    const z = ys.map((y) =>
      xs.map((_, j) => (curves.valid[2 * j] ? Math.exp(family.logPdf(y, curves.at(2 * j))) : NaN)),
    )
    return { xs, ys, z }
  }, [state.density, family, curves, yLimits])

  const invalid = curves.valid.filter((v) => !v).length / GRID.length
  const x = useAxis({ label: 'x', range: [0, 1] })
  // Fit, not hold: the axes follow the band and the curves as they are reshaped (frozen during a drag, refitted on release).
  const y = useAxis({ label: 'y' })
  const eta = useAxis({ label: 'link scale η_k(x)' })
  const band = curves.mean.map((m, i) => m + 2 * curves.sd[i])
  const bandLow = curves.mean.map((m, i) => m - 2 * curves.sd[i])

  return (
    <Figure
      title="GAMLSS as a generative model"
      state={state}
      caption={
        <>
          Each distribution parameter is a cubic B-spline in x on the link scale, η_k(x) = Σⱼ c_kj Bⱼ(x). The lower
          panel draws every η_k together, one colour per parameter; drag any control point c_kj up or down, and θ_k(x) =
          g_k⁻¹(η_k(x)) follows. The top panel shows the conditional distribution of y: the mean ± 2 standard deviations
          (band), the median (ink) and, as options, the 3rd to 97th centiles and the density. Dragging σ's points widens
          or narrows the band without moving the mean. Moving ν (Student t: tail weight; BCCG: skewness) changes the
          centiles' spacing. With a log link σ stays positive however far η₂ is dragged down; with the identity link the
          distribution is undefined wherever σ ≤ 0.
        </>
      }
      readouts={
        <>
          <Readout label="family" value={family.abbreviation} />
          <Readout label="share of x where θ is valid" value={`${Math.round(100 * (1 - invalid))}%`} />
        </>
      }
    >
      <Plot x={x} y={y} height={300}>
        {density && (
          <Raster name="density" x={density.xs} y={density.ys} z={density.z} colorBar={false} fillOpacity={0.8} />
        )}
        <Area name="mean ± 2 sd" x={GRID} y={band} base={bandLow} slot={0} opacity={0.15} />
        <Curve name="mean" x={GRID} y={curves.mean} slot={0} />
        <Curve name="median" x={GRID} y={curves.median} emphasis dashed />
        {state.centiles &&
          curves.centiles.map((c, j) => <Curve key={j} name="centiles 3–97" x={GRID} y={c} muted width={1} />)}
      </Plot>
      <Plot x={x} y={eta} height={240}>
        {family.parameters.map((p, k) => (
          <Curve
            key={p.name}
            name={`η for ${SYMBOL[p.name]} (${linkNames[k]} link)`}
            x={GRID}
            y={curves.eta[k]}
            slot={k}
          />
        ))}
        {family.parameters.map((p, k) => (
          <Points
            key={p.name}
            name={`η for ${SYMBOL[p.name]} (${linkNames[k]} link)`}
            x={CONTROL_X}
            y={control[k]}
            slot={k}
          />
        ))}
        {control.flatMap((c, k) =>
          c.map((v, j) => (
            <Handle key={`${k}-${j}`} kind="point" at={[CONTROL_X[j], v]} onDrag={([, w]) => move(k, j, w)} slot={k} />
          )),
        )}
      </Plot>
    </Figure>
  )
}
