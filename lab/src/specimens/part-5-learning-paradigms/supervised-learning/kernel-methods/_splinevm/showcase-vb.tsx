import { useMemo } from 'react'
import { Player, StatusText } from 'aifn-render/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, tex } from 'aifn-render/layout'
import { choice, float, int, row, toggle, useComputed, useFigureState, when } from 'aifn-render/state'
import { Area, Bars, Curve, Plot, Plots, Points, Readout, Rug, Segments, useAxis } from 'aifn-render/viz'
import { f3, usePlayed } from '../_gaussian-processes/incremental'
import {
  fitVb,
  LOSSES,
  makeData,
  normalCdf,
  predictVb,
  type DataSpec,
  type LossKey,
  type NoiseKind,
  type Task,
  type VbFit,
} from './vb'

const GRID = 241
const X = Array.from({ length: GRID }, (_, i) => i / (GRID - 1))
const INFORMATIVE = 0.5
/** ν at which the model is, for practical purposes, the plain GAM. */
const GAM_NU = 1000
const SWEEP_N = [25, 50, 100, 200, 400, 800]

const SCHEMA = {
  data: row('1 · data', {
    task: choice(
      [
        { value: 'reg', label: 'regression: wave + step' },
        { value: 'clf', label: 'classification: sine logit' },
        { value: 'count', label: 'counts: sine log-rate' },
      ],
      'reg',
      { label: 'problem' },
    ),
    amplitude: float(0.6, { ge: 0, le: 2, label: 'wave amplitude A', when: when('task', 'reg') }),
    cycles: float(2, { ge: 0.5, le: 6, label: 'wave cycles k', when: when('task', 'reg') }),
    stepHeight: float(1.5, { ge: -3, le: 3, label: 'step height H', when: when('task', 'reg') }),
    stepAt: float(0.6, { ge: 0.05, le: 0.95, label: 'step at c', when: when('task', 'reg') }),
    stepWidth: float(0.01, {
      gt: 0,
      scale: 'log10',
      suggestions: [0.003, 0.01, 0.03, 0.1],
      label: 'step width w',
      when: when('task', 'reg'),
    }),
    noiseKind: choice(
      [
        { value: 'gaussian', label: 'Gaussian' },
        { value: 'heavy', label: 'heavy-tailed (t₂)' },
        { value: 'outliers', label: '8% outliers' },
        { value: 'skewed', label: 'skewed (exponential)' },
      ],
      'gaussian',
      { label: 'noise', when: when('task', 'reg') },
    ),
    noise: float(0.15, { ge: 0.01, le: 1, label: 'noise scale', when: when('task', 'reg') }),
    mean: float(0, { ge: -4, le: 4, label: 'mean m (logit or log-rate)', when: (v) => v.task !== 'reg' }),
    linkAmplitude: float(3, { ge: 0, le: 8, label: 'amplitude A', when: (v) => v.task !== 'reg' }),
    linkCycles: float(2, { ge: 0.5, le: 6, label: 'cycles k', when: (v) => v.task !== 'reg' }),
    n: int(200, { ge: 20, le: 800, suggestions: [50, 200, 800], label: 'n' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  }),
  model: row('2 · the model', {
    loss: choice(
      LOSSES.filter((l) => l.value !== 'logistic' && l.value !== 'poisson'),
      'gaussian',
      { label: 'loss (as likelihood)', when: when('task', 'reg') },
    ),
    tau: float(0.9, {
      ge: 0.02,
      le: 0.98,
      suggestions: [0.1, 0.5, 0.9],
      label: 'quantile τ',
      when: (v) => v.task === 'reg' && v.loss === 'pinball',
    }),
    knots: int(30, { ge: 6, le: 60, suggestions: [15, 30, 50], label: 'knot grid M' }),
    adaptivity: float(0.5, {
      gt: 0,
      scale: 'log10',
      suggestions: [0.05, 0.5, 5, GAM_NU],
      label: 'ν: departure from the GAM (large = plain GAM)',
    }),
    learnKnots: toggle(false, 'learn knot locations'),
    leash: float(3, {
      ge: 0.25,
      le: 10,
      suggestions: [0.5, 1, 3, 5],
      label: 'knot leash (grid spacings)',
      when: (v) => Boolean(v.learnKnots),
    }),
  }),
  run: row('3 · fit', {
    maxIter: int(200, { ge: 5, le: 2000, suggestions: [50, 200, 500], label: 'iterations (at most)' }),
    sweep: toggle(false, 'complexity vs n: GAM against adaptive'),
  }),
}

/** Inverse of the standard normal cdf by bisection (display only). */
const normalQuantile = (p: number) => {
  let a = -10
  let b = 10
  for (let i = 0; i < 80; i++) {
    const c = (a + b) / 2
    if (normalCdf(c) < p) a = c
    else b = c
  }
  return (a + b) / 2
}

export function SplineVbShowcase() {
  const state = useFigureState(SCHEMA)
  const d = state.data
  const task = d.task as Task
  const loss: LossKey = task === 'clf' ? 'logistic' : task === 'count' ? 'poisson' : (state.model.loss as LossKey)
  const { tau, knots, adaptivity, learnKnots, leash } = state.model
  const knotLeash = learnKnots ? leash : 0
  const lossLabel = LOSSES.find((l) => l.value === loss)?.label ?? loss

  const spec: DataSpec = {
    task,
    amplitude: d.amplitude,
    cycles: d.cycles,
    stepHeight: d.stepHeight,
    stepAt: d.stepAt,
    stepWidth: d.stepWidth,
    noiseKind: d.noiseKind as NoiseKind,
    noise: d.noise,
    mean: d.mean,
    linkAmplitude: d.linkAmplitude,
    linkCycles: d.linkCycles,
    seed: d.seed,
  }
  const specKey = JSON.stringify(spec)
  const full = useMemo(() => makeData(JSON.parse(specKey) as DataSpec), [specKey])
  const data = useMemo(() => ({ xs: full.xs.slice(0, d.n), ys: full.ys.slice(0, d.n) }), [full, d.n])
  const truth = useMemo(() => X.map(full.truth), [full])

  // ── The fit (and the optional sweep over n), when a slider is let go ──
  const runKey = JSON.stringify([
    specKey,
    d.n,
    loss,
    tau,
    knots,
    adaptivity,
    knotLeash,
    state.run.maxIter,
    state.run.sweep,
  ])
  const computed = useComputed(
    () => {
      const opts = { loss, tau, knots, adaptivity, knotLeash, maxIter: state.run.maxIter }
      const fit = fitVb(data, opts)
      // The plain GAM on the same data: the adaptivity has paid for itself only if it raises the ELBO.
      const gam =
        adaptivity >= GAM_NU && knotLeash === 0 ? fit : fitVb(data, { ...opts, adaptivity: GAM_NU, knotLeash: 0 })
      const gamElbo = gam.states[gam.states.length - 1].elbo
      let sweep: null | { n: number[]; rows: { label: string; eff: number[]; bits: number[]; err: number[] }[] } = null
      if (state.run.sweep) {
        const variants = [
          { label: 'GAM (ν = 1000)', nu: GAM_NU },
          { label: `adaptive (ν = ${adaptivity >= 100 ? 0.5 : adaptivity})`, nu: adaptivity >= 100 ? 0.5 : adaptivity },
        ]
        sweep = {
          n: SWEEP_N,
          rows: variants.map((v) => {
            const eff: number[] = []
            const bits: number[] = []
            const err: number[] = []
            for (const n of SWEEP_N) {
              const f = fitVb({ xs: full.xs.slice(0, n), ys: full.ys.slice(0, n) }, { ...opts, adaptivity: v.nu })
              const st = f.states[f.states.length - 1]
              eff.push(st.gamma.reduce((a, b) => a + b, 0))
              bits.push(st.bits.reduce((a, b) => a + b, 0))
              const p = predictVb(f, st, X)
              err.push(Math.sqrt(X.reduce((a, _, i) => a + (f.loss.link(p.mean[i]) - truth[i]) ** 2, 0) / X.length))
            }
            return { label: v.label, eff, bits, err }
          }),
        }
      }
      return { key: runKey, fit, sweep, gamElbo }
    },
    [runKey],
    { mode: 'release' },
  )
  const fit: VbFit = computed.value.fit
  const sweep = computed.value.sweep
  const gamElbo = computed.value.gamElbo
  const [pos, setPos] = usePlayed(computed.value.key, fit.states.length)
  const st = fit.states[pos]

  // ── Views of the state at the played iteration ──
  const curve = useMemo(() => {
    const p = predictVb(fit, st, X)
    const link = fit.loss.link
    return {
      mean: p.mean.map(link),
      upper: p.mean.map((m, i) => link(m + 2 * Math.sqrt(p.variance[i]))),
      lower: p.mean.map((m, i) => link(m - 2 * Math.sqrt(p.variance[i]))),
      // y = f + noise, for the Gaussian loss: ± 2 sd of the curve plus the noise.
      yUpper:
        fit.loss.key === 'gaussian' ? p.mean.map((m, i) => m + 2 * Math.sqrt(p.variance[i] + st.scale ** 2)) : null,
      yLower:
        fit.loss.key === 'gaussian' ? p.mean.map((m, i) => m - 2 * Math.sqrt(p.variance[i] + st.scale ** 2)) : null,
    }
  }, [fit, st])
  const knotX = useMemo(() => st.knots.map((t) => fit.lo + t * fit.span), [fit, st])
  const gridX = useMemo(() => fit.knots.map((t) => fit.lo + t * fit.span), [fit])
  // Each knot's path over the iterations (learned locations only).
  const paths = useMemo(() => {
    if (knotLeash === 0) return null
    const segs: { from: [number, number]; to: [number, number] }[] = []
    for (let i = 1; i < fit.states.length; i++)
      fit.states[i].knots.forEach((t, j) =>
        segs.push({
          from: [fit.lo + fit.states[i - 1].knots[j] * fit.span, i - 1],
          to: [fit.lo + t * fit.span, i],
        }),
      )
    return segs
  }, [fit, knotLeash])
  const relevance = useMemo(() => {
    const gam = Math.log10(1 / Math.sqrt(st.gamLevel))
    const level = st.alpha.map((a) => Math.log10(1 / Math.sqrt(a)))
    const up = level.map((v, j) => ({ x: knotX[j], v })).filter((k) => k.v >= gam)
    const down = level.map((v, j) => ({ x: knotX[j], v })).filter((k) => k.v < gam)
    const stem = (k: { x: number; v: number }) => ({
      from: [k.x, gam] as [number, number],
      to: [k.x, k.v] as [number, number],
    })
    return { gam, up, down, upStems: up.map(stem), downStems: down.map(stem) }
  }, [st, knotX])
  const series = useMemo(
    () => ({
      i: fit.states.map((_, k) => k),
      elbo: fit.states.map((s) => s.elbo),
      eff: fit.states.map((s) => s.gamma.reduce((a, b) => a + b, 0)),
      bits: fit.states.map((s) => s.bits.reduce((a, b) => a + b, 0)),
    }),
    [fit],
  )
  // The true τ-quantile, where it has a closed form (Gaussian or exponential noise).
  const quantileTruth = useMemo(() => {
    if (task !== 'reg' || loss !== 'pinball') return null
    if (d.noiseKind === 'gaussian') return truth.map((f) => f + d.noise * normalQuantile(tau))
    if (d.noiseKind === 'skewed') return truth.map((f) => f + d.noise * (-Math.log(1 - tau) - 1))
    return null
  }, [task, loss, d.noiseKind, d.noise, tau, truth])

  const shownY = task === 'clf' ? data.ys.map((y, i) => y + 0.06 * (((i * 7919) % 101) / 101 - 0.5)) : data.ys
  const yRange: [number, number] =
    task === 'clf' ? [-0.08, 1.08] : [Math.min(...data.ys) - 0.3, Math.max(...data.ys) + 0.3]
  const xa = useAxis({ label: 'x', range: [0, 1] })
  const fa = useAxis({
    label: task === 'clf' ? 'P(y = 1)' : task === 'count' ? 'count (rate)' : 'y',
    range: yRange,
    key: `${task}-${d.noiseKind}`,
  })
  const ra = useAxis({ label: 'log₁₀ prior sd of w', key: computed.value.key })
  const ba = useAxis({ label: 'bits about w', range: [0, undefined], key: computed.value.key })
  const ia = useAxis({ label: 'iteration', range: [0, Math.max(1, fit.states.length - 1)], key: computed.value.key })
  const ea = useAxis({
    label: 'ELBO (nats)',
    range: [series.elbo[Math.min(1, series.elbo.length - 1)], undefined],
    key: computed.value.key,
  })
  const ka = useAxis({ label: 'Σγ (effective knots)', range: [0, undefined], key: computed.value.key })
  const pa = useAxis({ label: 'iteration', range: [0, Math.max(1, fit.states.length - 1)], key: computed.value.key })
  const na = useAxis({ label: 'log₁₀ n', range: [1.3, 3] })
  const sa = useAxis({ label: 'Σγ (effective knots)', range: [0, undefined], key: computed.value.key })
  const ra2 = useAxis({ label: 'RMS error of the fit', range: [0, undefined], key: computed.value.key })

  const nInformative = st.bits.filter((b) => b > INFORMATIVE).length
  // Calibration: how much of the truth lies inside the ± 2 sd band of f (for pinball, the true τ-quantile, if known).
  const target = loss === 'pinball' ? quantileTruth : truth
  const coverage = target
    ? target.filter(
        (t, i) => t >= Math.min(curve.lower[i], curve.upper[i]) && t <= Math.max(curve.lower[i], curve.upper[i]),
      ).length / target.length
    : NaN
  const eff = st.gamma.reduce((a, b) => a + b, 0)

  return (
    <Figure
      title="Variational GAM with adaptive smoothing, under any loss"
      purpose="A cubic-spline GAM term whose likelihood is any loss. Its smoothing is shared across knots unless the data pay, in evidence, for a knot to have more (or less) freedom: one knob, ν, prices that departure, from the plain GAM (large ν) to ARD-like sparsity (small ν)."
      state={state}
      defaultSize="XL"
      equation={
        <Equation>
          {tex`\begin{aligned} \kappa &\sim \mathrm{Gamma}(a_\kappa, b_\kappa) && \text{the GAM's shared smoothing level} \\ \alpha_j \mid \kappa &\sim \mathrm{Gamma}(\nu,\, \nu\kappa) && \text{each knot's relevance, } \mathbb E[\alpha_j \mid \kappa] = 1/\kappa \\ w_j \mid \alpha_j &\sim \mathcal N(0, \alpha_j^{-1}) && \text{jump in } f''' \text{ at grid knot } t_j \\ y_n &\sim p_\ell\big(y_n \mid f(x_n), s\big), \quad f(x) = \textstyle\sum_{k=0}^3 c_k u^k + \sum_j w_j (u - t_j)_+^3 && \text{any loss as a likelihood} \\ \text{ELBO} &= \textstyle\sum_n \mathbb E_q\big[\log p_\ell(y_n \mid f_n)\big] - \mathrm{KL}\big(q(\mathbf w, \boldsymbol\alpha, \kappa) \,\|\, p\big) && \text{fit} - \text{complexity} \end{aligned}`}
        </Equation>
      }
      controls={
        <Player
          label="4 · step through the variational iterations"
          value={pos}
          onChange={setPos}
          count={fit.states.length}
          format={(p) => {
            const s = fit.states[p]
            return s ? `iteration ${p} of ${fit.states.length - 1}: ELBO ${s.elbo.toFixed(1)}` : ''
          }}
        />
      }
      readouts={{
        [`iteration ${pos}`]: (
          <>
            <Readout label="ELBO" value={f3(st.elbo)} />
            <Readout
              label="ELBO − plain GAM’s"
              value={
                adaptivity >= GAM_NU && knotLeash === 0
                  ? '0 (this is the GAM)'
                  : f3(
                      fit.states[fit.states.length - 1].elbo - fit.states[fit.states.length - 1].knotLogPrior - gamElbo,
                    )
              }
            />
            {Number.isFinite(coverage) && (
              <Readout label="truth inside ± 2 sd of f" value={`${Math.round(100 * coverage)}%`} />
            )}
            <Readout label="expected fit Σ E[log p]" value={f3(st.fit)} />
            <Readout label="effective knots Σγ" value={f3(eff)} />
            <Readout
              label={`informative knots (> ${INFORMATIVE} bit)`}
              value={`${nInformative} of ${st.bits.length}`}
            />
            <Readout label="bits about the weights" value={f3(st.bits.reduce((a, b) => a + b, 0))} />
            {fit.loss.scaled && <Readout label="loss scale s" value={f3(st.scale)} />}
          </>
        ),
        run: (
          <>
            <Readout label="loss" value={lossLabel + (loss === 'pinball' ? `, τ = ${tau}` : '')} />
            <Readout label="iterations" value={fit.states.length - 1} />
            <Readout label="run time" value={`${Math.round(fit.ms)} ms`} />
            {computed.stale && <StatusText>computing…</StatusText>}
          </>
        ),
      }}
      caption={`${d.n} points; loss: ${lossLabel}${loss === 'pinball' ? ` at τ = ${tau}` : ''}, read as a likelihood with ${fit.loss.scaled ? 'a scale s fitted on the ELBO' : 'no scale'}. The spline has a cubic polynomial and one truncated cubic per knot, starting on a grid of ${knots}. ${knotLeash > 0 ? `The knot locations are learned on the ELBO plus a leash prior tⱼ ~ N(gⱼ, (${knotLeash} h)²) around each grid point gⱼ (h the spacing), so a knot can slide towards a feature but not wander off to chase single residuals; the knot paths are drawn below.` : 'The knots stay on the grid, so the flexible part cannot explain the data by putting a knot wherever a residual is; switch on “learn knot locations” to let them move on a leash.'} A plain GAM gives every knot the same prior precision (one smoothing level). Here each knot's precision αⱼ is drawn around that shared level with shape ν = ${adaptivity}: a knot departs from the GAM only when the evidence pays for it, and large ν recovers the GAM. Variational Bayes: q(w) Gaussian, q(α) and q(κ) Gamma (closed form); the loss enters through 1-D Gaussian expectations by quadrature, using loss values only, so non-smooth losses work. The fit starts at the MAP point and the ELBO rises at every iteration. Top: data, truth (ink, dashed${quantileTruth ? '; dotted, the true τ-quantile' : ''}), and the fit at this iteration with two bands: ± 2 sd of the curve f (darker; what the model is unsure of about f) and, for the Gaussian loss, ± 2 sd of a new y (lighter; the curve's uncertainty plus the noise s). The readouts check calibration (how much of the truth lies inside the f band; mean-field variational Bayes and ARD both tend to make it too narrow, and a feature narrower than the knot spacing cannot be covered by any band from this basis) and whether the adaptivity paid for itself (ELBO minus that of a plain GAM fitted to the same data). Middle left: each knot's prior sd 1/√E[αⱼ] against the GAM's shared level (dashed): stems up (orange) are knots given more freedom than the GAM, stems down are knots pushed towards zero. Below it, the bits learned about each knot's weight, ½ log₂(1/(E[αⱼ]Σⱼⱼ)). Middle right: the ELBO and the effective number of knots Σγ per iteration.${sweep ? ' Bottom: refits on nested subsamples, plain GAM against adaptive: how complexity and error follow n.' : ' Switch on “complexity vs n” to refit on nested subsamples and compare the plain GAM with the adaptive model.'}`}
    >
      <Dashboard>
        <DashboardRow ratio={1.2} minHeight={270}>
          <DashboardCell>
            <Plot x={xa} y={fa} title={`the fit at iteration ${pos}`}>
              {curve.yUpper && curve.yLower && (
                <Area
                  name="± 2 sd of y (curve + noise)"
                  x={X}
                  y={curve.yUpper}
                  base={curve.yLower}
                  slot={0}
                  opacity={0.08}
                  line={false}
                />
              )}
              <Area
                name="± 2 sd of f (the curve)"
                x={X}
                y={curve.upper}
                base={curve.lower}
                slot={0}
                opacity={0.25}
                line={false}
              />
              <Points name="data" x={data.xs} y={shownY} muted thin />
              <Curve name={task === 'reg' ? 'truth (location)' : 'truth'} x={X} y={truth} emphasis dashed />
              {quantileTruth && <Curve name="true τ-quantile" x={X} y={quantileTruth} slot={2} dashed />}
              <Curve name="fit" x={X} y={curve.mean} slot={0} />
              {knotLeash > 0 && <Rug name="starting grid" values={gridX} muted />}
              <Rug
                name={knotLeash > 0 ? 'knots (learned)' : 'knot grid'}
                values={knotX}
                slot={knotLeash > 0 ? 1 : undefined}
                muted={knotLeash === 0}
              />
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={300}>
          <DashboardCell ratio={1.4}>
            <Plots rows={2} tight>
              <Plot x={xa} y={ra} title="each knot's freedom against the GAM's shared level (dashed)">
                <Segments name="GAM level" segments={[{ from: [0, relevance.gam], to: [1, relevance.gam] }]} dashed />
                {relevance.upStems.length > 0 && (
                  <Segments name="more freedom" segments={relevance.upStems} slot={1} width={2} />
                )}
                {relevance.downStems.length > 0 && (
                  <Segments name="pushed to zero" segments={relevance.downStems} muted />
                )}
                {relevance.up.length > 0 && (
                  <Points
                    name="freer than the GAM"
                    x={relevance.up.map((k) => k.x)}
                    y={relevance.up.map((k) => k.v)}
                    slot={1}
                    size={8}
                  />
                )}
              </Plot>
              <Plot x={xa} y={ba} title="information: bits learned about each knot's weight">
                <Bars name="bits" x={knotX} y={st.bits} width={0.6 / knots} slot={1} />
                <Segments name="½ bit" segments={[{ from: [0, INFORMATIVE], to: [1, INFORMATIVE] }]} dashed />
              </Plot>
            </Plots>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={2} tight>
              <Plot x={ia} y={ea} title="ELBO = fit − complexity">
                <Curve name="ELBO" x={series.i} y={series.elbo} slot={0} />
                <Points name="now" x={[pos]} y={[series.elbo[pos]]} emphasis live />
              </Plot>
              <Plot x={ia} y={ka} title="effective knots Σγ">
                <Curve name="Σγ" x={series.i} y={series.eff} slot={1} />
                <Points name="now" x={[pos]} y={[series.eff[pos]]} emphasis live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
        {paths && (
          <DashboardRow minHeight={200}>
            <DashboardCell>
              <Plot x={xa} y={pa} title="learned knot locations over the iterations (dashed: this iteration)">
                <Segments name="knot paths" segments={paths} slot={1} width={1.5} />
                <Segments name="now" segments={[{ from: [0, pos], to: [1, pos] }]} dashed />
              </Plot>
            </DashboardCell>
          </DashboardRow>
        )}
        {sweep && (
          <DashboardRow minHeight={230}>
            <DashboardCell>
              <Plot x={na} y={sa} title="complexity follows the data: effective knots against n">
                {sweep.rows.map((r, i) => (
                  <Curve
                    key={r.label}
                    name={r.label}
                    x={sweep.n.map(Math.log10)}
                    y={r.eff}
                    slot={i === 0 ? 3 : 1}
                    showPoints
                  />
                ))}
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot x={na} y={ra2} title="error of the fit against n">
                {sweep.rows.map((r, i) => (
                  <Curve
                    key={r.label}
                    name={r.label}
                    x={sweep.n.map(Math.log10)}
                    y={r.err}
                    slot={i === 0 ? 3 : 1}
                    showPoints
                  />
                ))}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        )}
      </Dashboard>
    </Figure>
  )
}
