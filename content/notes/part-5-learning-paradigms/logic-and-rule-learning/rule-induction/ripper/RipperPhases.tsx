import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { simplifyRule, type Predicate } from 'aifn-compute/learning/explain'
import { stream } from 'aifn-compute/foundation/random'
import { applyRipper, ripper, type RipperStep, ruleText, thresholdConditions } from 'aifn-compute/learning/rules'
import { boxSegments, moonRows, MOONS_X, MOONS_Y } from '../_shared/rule-figures'

const PHASE: Record<RipperStep['phase'], string> = {
  grow: 'grow on the growing set (FOIL gain)',
  prune: 'prune on the pruning set ((p − n)/(p + n))',
  accept: 'accept; covered rows leave play',
  reject: 'reject and stop',
  delete: 'delete a rule that lengthens the description',
  optimise: 'optimise: original, replacement or revision',
  done: 'final rule set',
}

const show = (r: readonly Predicate[]) => ruleText(simplifyRule(r))

/** RIPPER on noisy two moons: growing, pruning, the description-length stop and the optimisation pass, stepped. */
export function RipperPhases() {
  const fig = useFigureState({
    noise: float(0.3, { min: 0, max: 0.5, step: 0.01, label: 'moons noise' }),
    optimisations: int(1, { min: 0, max: 4, label: 'optimisation passes k', suggestions: [0, 1, 2] }),
    growShare: float(0.67, { min: 0.4, max: 0.9, step: 0.01, label: 'growing share', suggestions: [0.5, 0.67, 0.8] }),
    seed: int(1, { min: 1, max: 999, label: 'seed' }),
  })
  const { noise, optimisations, growShare, seed } = fig

  const train = useMemo(() => moonRows(`ripper-${seed}`, 300, noise), [seed, noise])
  const test = useMemo(() => moonRows(`ripper-test-${seed}`, 2000, noise), [seed, noise])
  const result = useMemo(
    () =>
      ripper(train.rows, train.y, thresholdConditions(train.rows, 8), stream(`ripper-split-${seed}`), {
        positive: 1,
        optimisations,
        growFraction: growShare,
      }),
    [train, optimisations, growShare, seed],
  )
  const testAccuracy = useMemo(() => {
    const p = applyRipper(result, test.rows)
    return p.reduce((a, v, i) => a + (v === test.y[i] ? 1 : 0), 0) / test.y.length
  }, [result, test])

  const settings = `${noise}/${optimisations}/${growShare}/${seed}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const steps = result.trace
  const at = pos.settings === settings ? Math.min(pos.at, steps.length - 1) : 0
  const s = steps[at]

  // The rule set as it stands at this step: rules accepted and not since deleted or replaced.
  const ruleSet = useMemo(() => {
    let set: Predicate[][] = []
    for (let k = 0; k <= at; k++) {
      const t = steps[k]
      if (t.phase === 'accept') set = [...set, t.conditions]
      if (t.phase === 'delete') set = set.filter((r) => r !== t.conditions)
      if (t.phase === 'optimise') {
        const chosen = t.chosen === 'original' ? t.original : t.chosen === 'replacement' ? t.replacement : t.revision
        set = set.map((r) => (r === t.original ? chosen : r))
      }
      if (t.phase === 'done') set = t.rules
    }
    return set
  }, [steps, at])
  const dl = useMemo(
    () =>
      steps.slice(0, at + 1).flatMap((t, k) => (t.phase === 'accept' || t.phase === 'delete' ? [{ k, dl: t.dl }] : [])),
    [steps, at],
  )

  const split = s && 'split' in s ? s.split : null
  const lastSplit = useMemo(() => {
    for (let k = at; k >= 0; k--) {
      const t = steps[k]
      if ('split' in t) return t.split
    }
    return null
  }, [steps, at])
  const membership = split ?? lastSplit
  const idx = (v: number) => train.y.map((_, i) => i).filter((i) => (membership ? membership[i] === v : v === 1))
  const growRows = idx(1)
  const pruneRows = idx(2)
  const outRows = membership ? idx(0) : []

  const current: Predicate[] | null =
    s?.phase === 'grow' || s?.phase === 'accept' || s?.phase === 'reject'
      ? s.conditions
      : s?.phase === 'prune'
        ? s.candidates[s.kept].conditions
        : s?.phase === 'optimise'
          ? s.chosen === 'original'
            ? s.original
            : s.chosen === 'replacement'
              ? s.replacement
              : s.revision
          : null

  const xa = useAxis({ label: 'x₁', range: MOONS_X, nice: false })
  const ya = useAxis({ label: 'x₂', range: MOONS_Y, nice: false, equal: xa })
  const xb = useAxis({ label: 'trace step', range: [0, Math.max(1, steps.length - 1)], integer: true })
  const yb = useAxis({ label: 'description length (bits)' })
  if (!s) return null
  const pts = (rows: number[]) => ({ x: rows.map((i) => train.x1[i]), y: rows.map((i) => train.x2[i]) })

  return (
    <Figure
      title="RIPPER: grow, prune, stop by description length"
      state={fig}
      caption="RIPPER learning rules for class 1 (the inner moon) on 300 noisy two-moons points. For each rule the rows in play are split at random: circles are the growing set, diamonds the pruning set, grey points have left play. The rule grows on the growing set one threshold at a time, choosing the condition of largest FOIL gain, until it covers no class-0 growing point; then its final conditions are deleted while that raises (p − n)/(p + n) on the pruning set. The ink box is the rule at the current step and the grey boxes the rules kept so far. Right: the description length of the rule set and its errors after each accepted or deleted rule; RIPPER stops once it exceeds the smallest value by 64 bits or a pruned rule is wrong on half the pruning set, then deletes rules that lengthen the description, then runs k optimisation passes. Step through with the player; raise the noise to see rules grown deep and pruned back."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={steps.length} label="step" />}
      readouts={
        <>
          <Readout label="phase" value={PHASE[s.phase]} />
          {current && <Readout label="rule" value={`if ${show(current)} then class 1`} />}
          {s.phase === 'grow' && (
            <>
              <Readout label="growing set (class 1 : class 0)" value={`${s.grow.p} : ${s.grow.n}`} />
              <Readout label="pruning set (class 1 : class 0)" value={`${s.prune.p} : ${s.prune.n}`} />
            </>
          )}
          {s.phase === 'prune' && (
            <Readout
              label="pruning value of the rule and each shorter prefix"
              value={s.candidates
                .map(
                  (c, k) =>
                    `${k === s.kept ? '▸ ' : ''}${c.conditions.length} cond: ${Number.isFinite(c.value) ? formatNumber(c.value) : 'covers none'}`,
                )
                .join('; ')}
            />
          )}
          {(s.phase === 'accept' || s.phase === 'reject') && (
            <Readout
              label="description length, smallest so far"
              value={`${formatNumber(s.dl)}, ${formatNumber(s.minDl)} bits${s.reason ? ` (${s.reason})` : ''}`}
            />
          )}
          {s.phase === 'optimise' && (
            <Readout
              label="DL with original, replacement, revision"
              value={`${formatNumber(s.dls.original)}, ${formatNumber(s.dls.replacement)}, ${formatNumber(s.dls.revision)} → ${s.chosen}`}
            />
          )}
          <Readout label="rules now" value={ruleSet.length} />
          <Readout label="test accuracy of the final rules" value={formatNumber(testAccuracy)} />
        </>
      }
    >
      <Plots cols={2} widths={[60, 40]}>
        <Plot x={xa} y={ya}>
          {outRows.length > 0 && <Points name="out of play" {...pts(outRows)} muted thin />}
          {growRows.length > 0 && (
            <Points
              name="growing set"
              {...pts(growRows)}
              group={growRows.map((i) => train.y[i])}
              groupNames={['class 0', 'class 1']}
              shape={0}
            />
          )}
          {pruneRows.length > 0 && (
            <Points
              name="pruning set"
              {...pts(pruneRows)}
              group={pruneRows.map((i) => train.y[i])}
              groupNames={['class 0', 'class 1']}
              shape={3}
            />
          )}
          {ruleSet.length > 0 && (
            <Segments
              name="rules kept"
              segments={ruleSet.flatMap((r) => boxSegments(r, MOONS_X, MOONS_Y))}
              muted
              width={1.2}
            />
          )}
          {current && current.length > 0 && (
            <Segments name="current rule" segments={boxSegments(current, MOONS_X, MOONS_Y)} emphasis width={2.5} />
          )}
        </Plot>
        <Plot x={xb} y={yb}>
          {dl.length > 0 && (
            <Curve name="description length" x={dl.map((d) => d.k)} y={dl.map((d) => d.dl)} showPoints slot={2} />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
