import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { linearRegressionData } from 'aifn-methods/data/synthetic'
import { elasticNetPath, groupLassoPath } from 'aifn-methods/learning/linear'

// Three groups: a three-column factor with signal, a two-column group of noise, and a two-column group with weak signal.
const GROUPS = [0, 0, 0, 1, 1, 2, 2]
const GROUP_NAMES = ['group A (signal, 3 columns)', 'group B (noise, 2 columns)', 'group C (weak, 2 columns)']
const TRUE_W = [1, -0.8, 0.4, 0, 0, 0.5, 0.2]
const LOG_ALPHAS = Array.from({ length: 71 }, (_, i) => -3 + i * 0.05)
const ALPHAS = LOG_ALPHAS.map((l) => 10 ** l)
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** The group-lasso and lasso paths of the same data, coefficients coloured by group (`groupLassoPath`, `elasticNetPath`). */
export function GroupLassoPaths() {
  const state = useFigureState({
    n: int(60, { ge: 20, le: 500, suggestions: [30, 60, 200], label: 'data points n' }),
    correlation: slider(-0.9, 0.9, 0.5, { step: 0.05, label: 'neighbouring-feature correlation ρ' }),
    noise: float(1, { min: 0.2, max: 3, step: 0.1, label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    logAlpha: float(-1, { min: -3, max: 0.5, step: 0.05, label: 'log₁₀ penalty α' }),
  })
  const { n, correlation, noise, seed } = state
  const data = useMemo(
    () => linearRegressionData(stream(`group-lasso/${seed}`), { n, weights: TRUE_W, noise, correlation }),
    [n, correlation, noise, seed],
  )
  const run = useComputed(
    () => {
      const y = data.y!
      // Both paths are fitted from the largest penalty down; reverse to the increasing order of ALPHAS.
      const group = groupLassoPath(data.x, y, { groups: GROUPS, alphas: ALPHAS })
      const lasso = elasticNetPath(data.x, y, { alphas: ALPHAS })
      const rev = (rows: number[][]) => rows.map((r) => [...r].reverse())
      return {
        group: rev(toRows(group.coef)),
        lasso: rev(toRows(lasso.coef)),
        active: [...group.activeGroups].reverse(),
        nonzero: [...lasso.nonzero].reverse(),
      }
    },
    [data],
    { mode: 'release' },
  )
  const v = run.value
  const j = Math.max(0, Math.min(ALPHAS.length - 1, Math.round((state.logAlpha + 3) / 0.05)))
  const zeroInside = (coef: number[][]) =>
    [0, 1, 2].filter((g) => {
      const members = GROUPS.map((h, c) => (h === g ? coef[c][j] : null)).filter((x) => x !== null) as number[]
      return members.some((x) => x === 0) && members.some((x) => x !== 0)
    }).length

  const ax = useAxis({ label: 'log₁₀ penalty α', range: [-3, 0.5], nice: false })
  const gy = useAxis({ label: 'coefficient', hold: 'union', key: `${n}/${correlation}/${noise}/${seed}` })
  const setAlpha = (x: number) => state.set('logAlpha', Math.max(-3, Math.min(0.5, Math.round(x * 20) / 20)))
  const panel = (coef: number[][], title: string) => (
    <Plot x={ax} y={gy} title={title}>
      {coef.map((row, c) => (
        <Curve
          key={c}
          id={`c${c}`}
          name={GROUP_NAMES[GROUPS[c]]}
          x={LOG_ALPHAS}
          y={row}
          slot={GROUPS[c]}
          stale={run.stale}
        />
      ))}
      <Handle kind="x" at={LOG_ALPHAS[j]} label="α" onDrag={setAlpha} />
    </Plot>
  )

  return (
    <Figure
      title="Group lasso against lasso"
      state={state}
      hoverReadout={false}
      caption="Seven standard normal features in three groups, with true coefficients (1, −0.8, 0.4) on group A, zero on group B and (0.5, 0.2) on group C, and correlation ρ between neighbouring features. Left: the group-lasso path, ‖y − Xw‖² / 2n + α Σ √p_g ‖w_g‖₂; right: the lasso path, ‖y − Xw‖² / 2n + α‖w‖₁, of the same data, both from the engine. Colours mark the groups. In the group lasso every coefficient of a group leaves zero at the same penalty, and a group is either wholly in or wholly out; in the lasso the members of a group enter one at a time, so a factor can be half in the model. Drag the vertical line in either panel to move α."
      readouts={
        <>
          <Readout label="α" value={fmt(ALPHAS[j])} />
          <Readout label="group lasso: groups in the model" value={`${v.active[j]} of 3`} />
          <Readout label="lasso: non-zero coefficients" value={`${v.nonzero[j]} of 7`} />
          <Readout
            label="groups partly zero (group lasso / lasso)"
            value={`${zeroInside(v.group)} / ${zeroInside(v.lasso)}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {panel(v.group, 'group lasso')}
        {panel(v.lasso, 'lasso')}
      </div>
    </Figure>
  )
}
