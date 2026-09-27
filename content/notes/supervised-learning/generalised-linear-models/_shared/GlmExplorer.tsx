import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { FAMILIES, LINKS, irls, simulate, type FamilyName, type LinkName } from './glm'

const N = 100
const GRID = linspace(-2, 2, 81)

const FAMILY_OPTIONS = (Object.keys(FAMILIES) as FamilyName[]).map((f) => ({ value: f, label: FAMILIES[f].label }))
const LINK_LABELS: Record<LinkName, string> = {
  identity: 'identity',
  log: 'log',
  logit: 'logit',
  cloglog: 'comp. log-log',
  inverse: 'inverse',
  sqrt: 'square root',
}

export function GlmExplorer({ initialFamily = 'poisson' }: { initialFamily?: FamilyName }) {
  const [familyName, setFamilyName] = useState<FamilyName>(initialFamily)
  const [linkName, setLinkName] = useState<LinkName>(FAMILIES[initialFamily].canonical)
  const [seed, setSeed] = useState(1)
  const [shown, setShown] = useState(99)
  const [showPath, setShowPath] = useState(true)
  const family = FAMILIES[familyName]

  const data = useMemo(() => simulate(familyName, N, rng(seed * 101 + familyName.length)), [familyName, seed])
  const fit = useMemo(() => irls(data.x, data.y, family, LINKS[linkName]), [data, family, linkName])
  const last = fit.iterates.length - 1
  const k = Math.min(shown, last)
  const link = LINKS[linkName]
  const curve = (b: [number, number]) => GRID.map((g) => link.inv(b[0] + b[1] * g))

  const yMax = Math.max(...data.y)
  const yRange: [number, number] =
    familyName === 'binomial' ? [-0.05, 1.05] : [Math.min(0, Math.min(...data.y)), Math.ceil(yMax * 1.05)]
  const clip = (v: number) => Math.min(Math.max(v, yRange[0]), yRange[1])

  const series: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, slot: 0 },
    { name: 'true mean', type: 'line', x: GRID, y: GRID.map(data.truth), slot: 2, dashed: true },
    { name: `fitted mean, iterate ${k}`, type: 'line', x: GRID, y: curve(fit.iterates[k].beta).map(clip), slot: 1 },
  ]

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

  return (
    <Interactive
      title="A GLM fitted by IRLS"
      caption={
        <>
          One hundred points drawn from the chosen family, with the true mean dashed. Each IRLS iterate solves a
          weighted least-squares problem on the working response. Step through the iterates: iterate 0 is the constant
          fit at the sample mean. The canonical link reaches the maximum in a handful of steps; the Gaussian family with
          the identity link needs one. A non-canonical link fits a different mean curve to the same data.
        </>
      }
      controls={
        <>
          <ParamChoice
            label="family"
            value={familyName}
            onChange={(f) => {
              setFamilyName(f)
              setLinkName(FAMILIES[f].canonical)
            }}
            options={FAMILY_OPTIONS}
          />
          <ParamChoice
            label="link"
            value={linkName}
            onChange={setLinkName}
            options={family.links.map((l) => ({
              value: l,
              label: l === family.canonical ? `${LINK_LABELS[l]} (canonical)` : LINK_LABELS[l],
            }))}
          />
          <ParamSlider
            label="iterate"
            value={k}
            onChange={setShown}
            min={0}
            max={Math.max(last, 1)}
            step={1}
            withArrows
          />
          <ParamSwitch label="show earlier iterates" checked={showPath} onChange={setShowPath} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="β₀" value={formatNumber(fit.iterates[k].beta[0])} />
          <Readout label="β₁" value={formatNumber(fit.iterates[k].beta[1])} />
          <Readout label="deviance" value={formatNumber(fit.iterates[k].deviance)} />
          <Readout label="iterations to converge" value={fit.converged ? String(last) : `> ${last}`} />
        </>
      }
    >
      <XYChart series={series} segments={path} xRange={[-2, 2]} yRange={yRange} xLabel="x" yLabel="y" />
    </Interactive>
  )
}
