import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
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
  const state = useFigureState({
    rank: int(2, { min: 1, max: 5, step: 1, label: 'latent dimensions d', format: (v) => String(v) }),
    lambda: float(1, { min: 0.05, max: 4, step: 0.05, label: 'regularisation λ' }),
    sweep: int(STEPS, { min: 0, max: STEPS, step: 1, label: 'ALS sweeps', format: (v) => String(v) }),
  })

  const hist = useMemo(() => history(DATA, state.rank, state.lambda, STEPS), [state.rank, state.lambda])
  const curves = useMemo(() => {
    const tr = hist.map((f) => rmse(DATA, f, DATA.train))
    const te = hist.map((f) => rmse(DATA, f, DATA.test))
    return [
      { name: 'training RMSE', x: SWEEPS, y: tr, slot: 0 },
      { name: 'held-out RMSE', x: SWEEPS, y: te, slot: 1 },
    ] as const
  }, [hist])
  const f = hist[state.sweep]
  const z = useMemo(() => USERS.map((_, u) => ITEMS.map((_, i) => predict(DATA, f.W[u], f.V[i]))), [f])

  const xAxis = useAxis({ label: 'item' })
  const yAxis = useAxis({ label: 'user' })
  const xAxis2 = useAxis({ label: 'ALS sweep', range: [0, STEPS] })
  const yAxis2 = useAxis({ label: 'RMSE', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Filling in a ratings matrix"
      state={state}
      caption="Twelve users rate sixteen items on a 1–5 scale. The true ratings come from two hidden taste dimensions plus noise. Markers show which ratings were used for training and which were held out; every other cell is predicted by r̂ = mean + w_u · v_i. Step through ALS sweeps with the arrows. Rank 1 underfits; a high rank with little regularisation fits the training ratings closely and predicts the held-out ones worse. A large λ shrinks every prediction towards the mean."

      readouts={
        <>
          <Readout label="training RMSE" value={formatNumber(curves[0].y[state.sweep])} />
          <Readout label="held-out RMSE" value={formatNumber(curves[1].y[state.sweep])} />
          <Readout
            label="observed cells"
            value={`${DATA.train.length + DATA.test.length} of ${DATA.users * DATA.items}`}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Raster x={ITEMS} y={USERS} z={z} range={[1, 5]} valueLabel={'predicted rating'} />
          <Points name="training rating" x={TRAIN_X} y={TRAIN_Y} live />
          <Points name="held-out rating" x={TEST_X} y={TEST_Y} live />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...curves[0]} />
          <Curve {...curves[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
