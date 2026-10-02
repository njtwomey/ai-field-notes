import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { history, makeRatings, predict, rmse } from '../_shared/mf'

const DATA = makeRatings()
const STEPS = 20
const USERS = Array.from({ length: DATA.users }, (_, u) => u + 1)
const ITEMS = Array.from({ length: DATA.items }, (_, i) => i + 1)
const TRAIN_X = DATA.train.map((t) => t[1] + 1)
const TRAIN_Y = DATA.train.map((t) => t[0] + 1)
const TEST_X = DATA.test.map((t) => t[1] + 1)
const TEST_Y = DATA.test.map((t) => t[0] + 1)
const SWEEPS = Array.from({ length: STEPS + 1 }, (_, t) => t)

/** Matrix factorisation of a small ratings matrix by ALS: rank, regularisation and the number of sweeps. */
export function MfFigure() {
  const rank = useParam(2, { min: 1, max: 5, step: 1 })
  const lambda = useParam(1, { min: 0.05, max: 4, step: 0.05 })
  const sweep = useParam(STEPS, { min: 0, max: STEPS, step: 1 })

  const hist = useMemo(() => history(DATA, rank.value, lambda.value, STEPS), [rank.value, lambda.value])
  const curves = useMemo((): XYSeries[] => {
    const tr = hist.map((f) => rmse(DATA, f, DATA.train))
    const te = hist.map((f) => rmse(DATA, f, DATA.test))
    return [
      { name: 'training RMSE', type: 'line', x: SWEEPS, y: tr, slot: 0 },
      { name: 'held-out RMSE', type: 'line', x: SWEEPS, y: te, slot: 1 },
    ]
  }, [hist])
  const f = hist[sweep.value]
  const z = useMemo(() => USERS.map((_, u) => ITEMS.map((_, i) => predict(DATA, f.W[u], f.V[i]))), [f])

  return (
    <Interactive
      title="Filling in a ratings matrix"
      caption="Twelve users rate sixteen items on a 1–5 scale. The true ratings come from two hidden taste dimensions plus noise. Markers show which ratings were used for training and which were held out; every other cell is predicted by r̂ = mean + w_u · v_i. Step through ALS sweeps with the arrows. Rank 1 underfits; a high rank with little regularisation fits the training ratings closely and predicts the held-out ones worse. A large λ shrinks every prediction towards the mean."
      controls={
        <>
          <ParamSlider label="latent dimensions d" param={rank} format={(v) => String(v)} />
          <ParamSlider label="regularisation λ" param={lambda} />
          <ParamSlider label="ALS sweeps" param={sweep} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="training RMSE" value={formatNumber(curves[0].y[sweep.value])} />
          <Readout label="held-out RMSE" value={formatNumber(curves[1].y[sweep.value])} />
          <Readout
            label="observed cells"
            value={`${DATA.train.length + DATA.test.length} of ${DATA.users * DATA.items}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Heatmap
          x={ITEMS}
          y={USERS}
          z={z}
          range={[1, 5]}
          xLabel="item"
          yLabel="user"
          valueLabel="predicted rating"
          overlay={[
            { name: 'training rating', type: 'scatter', x: TRAIN_X, y: TRAIN_Y },
            { name: 'held-out rating', type: 'scatter', x: TEST_X, y: TEST_Y },
          ]}
          height={320}
        />
        <XYChart
          series={curves}
          xLabel="ALS sweep"
          yLabel="RMSE"
          xRange={[0, STEPS]}
          yRange={[0, undefined]}
          height={320}
        />
      </div>
    </Interactive>
  )
}
