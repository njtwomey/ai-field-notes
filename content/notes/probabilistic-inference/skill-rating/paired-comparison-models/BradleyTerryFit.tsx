import { Minus, Plus } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { bradleyTerryLogLik, bradleyTerryMM } from '../_shared/skill'

const NAMES = ['A', 'B', 'C', 'D']
const MAX_ITER = 25
const START = [
  [0, 2, 3, 1],
  [1, 0, 2, 2],
  [0, 1, 0, 2],
  [1, 1, 1, 0],
]

/** True when every player can reach every other along "beat" edges: Ford's condition for a finite MLE. */
function stronglyConnected(wins: number[][]): boolean {
  const m = wins.length
  const reach = wins.map((row, i) => row.map((w, j) => i === j || w > 0))
  for (let k = 0; k < m; k++)
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) reach[i][j] ||= reach[i][k] && reach[k][j]
  return reach.every((row) => row.every(Boolean))
}

export function BradleyTerryFit() {
  const [wins, setWins] = useState(START)
  const iter = useParam(3, { min: 0, max: MAX_ITER, step: 1 })

  const fit = useMemo(() => {
    const history = bradleyTerryMM(wins, MAX_ITER)
    return { history, ll: history.map((th) => bradleyTerryLogLik(wins, th)), ok: stronglyConnected(wins) }
  }, [wins])

  const k = iter.value
  const iters = useMemo(() => Array.from({ length: MAX_ITER + 1 }, (_, i) => i), [])
  const series = useMemo<XYSeries[]>(
    () => [
      ...NAMES.map<XYSeries>((name, p) => ({
        name,
        type: 'line',
        x: iters,
        y: fit.history.map((th) => th[p]),
        slot: p,
      })),
      {
        name: `iteration ${k}`,
        type: 'scatter',
        x: NAMES.map(() => k),
        y: fit.history[k],
        emphasis: true,
      },
    ],
    [fit, iters, k],
  )
  const handles: Handle[] = [{ kind: 'x', at: k, label: 'iteration', onDrag: iter.set }]

  const change = (i: number, j: number, delta: number) =>
    setWins((w) => w.map((row, a) => row.map((v, b) => (a === i && b === j ? Math.max(0, v + delta) : v))))

  const theta = fit.history[k]
  const pAB = 1 / (1 + Math.exp(theta[1] - theta[0]))

  return (
    <Interactive
      title="Fitting Bradley–Terry by Zermelo's iteration"
      caption="Each cell counts the wins of the row player over the column player; change them with the buttons. The chart shows the log-strengths θ after each MM iteration, centred to mean zero. Drag the vertical line or use the slider to pick an iteration. The log-likelihood rises at every step. Make a player lose every game, or win every game, and the strengths drift apart without converging: the maximum-likelihood estimate no longer exists."
      controls={
        <>
          <ParamSlider label="iteration" param={iter} format={(v) => String(v)} withArrows />
          <ParamButton onClick={() => setWins(START)}>Reset results</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="log-likelihood at this iteration" value={formatNumber(fit.ll[k])} />
          <Readout label={`after ${MAX_ITER}`} value={formatNumber(fit.ll[MAX_ITER])} />
          <Readout label="P(A beats B)" value={formatNumber(pAB)} />
          <Readout label="finite MLE" value={fit.ok ? 'yes' : 'no: some group never loses to the rest'} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <table className="mx-auto text-xs tabular-nums">
          <thead>
            <tr>
              <th className="px-1 text-left font-normal text-muted-foreground">beat →</th>
              {NAMES.map((n) => (
                <th key={n} className="px-1 font-medium">
                  {n}
                </th>
              ))}
              <th className="px-1 font-normal text-muted-foreground">wins</th>
            </tr>
          </thead>
          <tbody>
            {wins.map((row, i) => (
              <tr key={NAMES[i]}>
                <th className="px-1 text-left font-medium">{NAMES[i]}</th>
                {row.map((v, j) => (
                  <td key={NAMES[j]} className="px-0.5 py-0.5">
                    {i === j ? (
                      <span className="block text-center text-muted-foreground">–</span>
                    ) : (
                      <span className="flex items-center gap-0.5">
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label={`one fewer win for ${NAMES[i]} over ${NAMES[j]}`}
                          disabled={v === 0}
                          onClick={() => change(i, j, -1)}
                        >
                          <Minus />
                        </Button>
                        <span className="w-3 text-center font-mono">{v}</span>
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-label={`one more win for ${NAMES[i]} over ${NAMES[j]}`}
                          disabled={v >= 9}
                          onClick={() => change(i, j, 1)}
                        >
                          <Plus />
                        </Button>
                      </span>
                    )}
                  </td>
                ))}
                <td className="px-1 text-center font-mono">{row.reduce((a, b) => a + b, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <XYChart
          series={series}
          xLabel="MM iteration"
          yLabel="log-strength θ"
          xRange={[0, MAX_ITER]}
          yRange={[-2.5, 2.5]}
          height={280}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
