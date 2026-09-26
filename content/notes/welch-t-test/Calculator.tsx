import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, formatNumber } from '@/components/viz'
import { studentTCdf } from '@/lib/math/special'

type Result = { t: number; df: number; p: number }

const twoSided = (t: number, df: number) => 2 * (1 - studentTCdf(Math.abs(t), df))

/** Both tests from summary statistics, updated live. */
export function Calculator() {
  const [diff, setDiff] = useState(1)
  const [sd1, setSd1] = useState(1)
  const [sd2, setSd2] = useState(3)
  const [n1, setN1] = useState(40)
  const [n2, setN2] = useState(10)

  const { pooled, welch } = useMemo(() => {
    const v1 = sd1 * sd1
    const v2 = sd2 * sd2
    const sp = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    const tp = diff / Math.sqrt(sp * (1 / n1 + 1 / n2))
    const a = v1 / n1
    const b = v2 / n2
    const tw = diff / Math.sqrt(a + b)
    const dfw = (a + b) ** 2 / ((a * a) / (n1 - 1) + (b * b) / (n2 - 1))
    const pooled: Result = { t: tp, df: n1 + n2 - 2, p: twoSided(tp, n1 + n2 - 2) }
    const welch: Result = { t: tw, df: dfw, p: twoSided(tw, dfw) }
    return { pooled, welch }
  }, [diff, sd1, sd2, n1, n2])

  const row = (label: string, r: Result) => (
    <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 rounded-md border px-4 py-3">
      <span className="w-32 text-sm font-medium">{label}</span>
      <Readout label="t" value={formatNumber(r.t)} />
      <Readout label="df" value={formatNumber(r.df)} />
      <Readout label="p" value={formatNumber(r.p)} />
      <span className={r.p < 0.05 ? 'text-xs font-medium' : 'text-xs text-muted-foreground'}>
        {r.p < 0.05 ? 'reject at α = 0.05' : 'do not reject'}
      </span>
    </div>
  )

  return (
    <Interactive
      title="Same summary statistics, two tests"
      caption="Enter each group's size and standard deviation and the observed difference in means. When the smaller group has the larger spread, the pooled test understates the standard error and reports a p-value that is too small."
      controls={
        <>
          <ParamSlider label="difference in means" value={diff} onChange={setDiff} min={0} max={3} step={0.05} />
          <ParamSlider label="sd, group 1" value={sd1} onChange={setSd1} min={0.2} max={5} step={0.1} />
          <ParamSlider label="sd, group 2" value={sd2} onChange={setSd2} min={0.2} max={5} step={0.1} />
          <ParamSlider label="n, group 1" value={n1} onChange={setN1} min={2} max={100} step={1} />
          <ParamSlider label="n, group 2" value={n2} onChange={setN2} min={2} max={100} step={1} />
        </>
      }
    >
      <div className="space-y-2">
        {row("Pooled (Student's)", pooled)}
        {row("Welch's", welch)}
      </div>
    </Interactive>
  )
}
