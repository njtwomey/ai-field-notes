import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  setting,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { FAMILIES, LINKS, irls, simulate, type FamilyName, type LinkName } from './glm'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const N = 100
const GRID = toFlat(linspace(-2, 2, 81))

const LINK_LABELS: Record<LinkName, string> = {
  identity: 'identity',
  log: 'log',
  logit: 'logit',
  cloglog: 'comp. log-log',
  inverse: 'inverse',
  sqrt: 'square root',
}

/** Each family with its own choice of link, opening on the canonical one. */
const schema = (initialFamily: FamilyName) => ({
  family: variants(
    Object.fromEntries(
      (Object.keys(FAMILIES) as FamilyName[]).map((f) => {
        const family = FAMILIES[f]
        const options = family.links.map((l) => ({
          value: l,
          label: l === family.canonical ? `${LINK_LABELS[l]} (canonical)` : LINK_LABELS[l],
        }))
        return [
          f,
          { label: family.label, params: { link: choice<LinkName>(options, family.canonical, { label: 'link' }) } },
        ]
      }),
    ) as Record<FamilyName, { label: string; params: { link: ReturnType<typeof choice<LinkName>> } }>,
    { label: 'Model', choiceLabel: 'family', initial: initialFamily },
  ),
  showPath: setting(true, 'show earlier iterates'),
  seed: int(1, { ge: 0, label: 'seed' }),
})

/** Seeded draws for `simulate`. */
function draws(seed: number) {
  const g = stream(seed)
  return { uniform: () => uniform(g), normal: () => normal(g) }
}

export function GlmExplorer({ initialFamily = 'poisson' }: { initialFamily?: FamilyName }) {
  const state = useFigureState(schema(initialFamily))
  const familyName = state.family.key as FamilyName
  const linkName = state.family.values.link as LinkName
  const { seed, showPath } = state
  const family = FAMILIES[familyName]

  const data = useMemo(() => simulate(familyName, N, draws(seed * 101 + familyName.length)), [familyName, seed])
  const fit = useMemo(() => irls(data.x, data.y, family, LINKS[linkName]), [data, family, linkName])
  const last = fit.iterates.length - 1
  // The walk through the iterates restarts at iterate 0 for a new fit.
  const [position, setPosition] = useState({ fit, k: 0 })
  const k = position.fit === fit ? Math.min(position.k, last) : 0
  const link = LINKS[linkName]
  const curve = (b: [number, number]) => GRID.map((g) => link.inv(b[0] + b[1] * g))

  const yMax = Math.max(...data.y)
  const yRange: [number, number] =
    familyName === 'binomial' ? [-0.05, 1.05] : [Math.min(0, Math.min(...data.y)), Math.ceil(yMax * 1.05)]
  const clip = (v: number) => Math.min(Math.max(v, yRange[0]), yRange[1])

  const series = [
    { name: 'data', x: data.x, y: data.y, slot: 0 },
    { name: 'true mean', x: GRID, y: GRID.map(data.truth), slot: 2, dashed: true },
    { name: `fitted mean, iterate ${k}`, x: GRID, y: curve(fit.iterates[k].beta).map(clip), slot: 1 },
  ] as const

  // Earlier iterates as thin muted polylines, so they add no legend entries.
  const path: Segment[] = showPath
    ? fit.iterates.slice(0, k).flatMap((it) => {
        const c = curve(it.beta).map(clip)
        return GRID.slice(1).map((g, i) => ({
          from: [GRID[i], c[i]] as [number, number],
          to: [g, c[i + 1]] as [number, number],
        }))
      })
    : []

  const xAxis = useAxis({ label: 'x', range: [-2, 2] })
  const yAxis = useAxis({ label: 'y', range: yRange })
  return (
    <Figure
      title="A GLM fitted by IRLS"
      state={state}
      caption={
        <>
          One hundred points drawn from the chosen family, with the true mean dashed. Each IRLS iterate solves a
          weighted least-squares problem on the working response. Step through the iterates: iterate 0 is the constant
          fit at the sample mean. The canonical link reaches the maximum in a handful of steps; the Gaussian family with
          the identity link needs one. A non-canonical link fits a different mean curve to the same data.
        </>
      }
      controls={
        <Player
          value={k}
          onChange={(v) => setPosition({ fit, k: v })}
          count={last + 1}
          label="iterate"
          format={(v) => `iterate ${v}`}
        />
      }
      readouts={
        <>
          <Readout label="β₀" value={formatNumber(fit.iterates[k].beta[0])} />
          <Readout label="β₁" value={formatNumber(fit.iterates[k].beta[1])} />
          <Readout label="deviance" value={formatNumber(fit.iterates[k].deviance)} />
          <Readout label="iterations to converge" value={fit.converged ? String(last) : `> ${last}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Segments segments={path} />
      </Plot>
    </Figure>
  )
}
