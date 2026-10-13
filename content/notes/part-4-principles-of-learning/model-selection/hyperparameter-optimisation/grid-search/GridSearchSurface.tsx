import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { grid, real, space } from 'aifn-compute/foundation/space'
import { tensor } from 'aifn-compute/foundation/tensor'
import { svmTuningSurface } from 'aifn-methods/data/objectives'
import { SVM_BEST, SVM_HI, SVM_LO, SVM_SPACE, svmError, svmRaster } from '../_shared/svmSurface'

type Point = { logC: number; logGamma: number }

/**
 * Grid search on the tabulated cross-validated error of an RBF SVM: a coarse grid, an optional fine grid around its
 * winner, and optional Gaussian noise on every score. The winner is chosen by the noisy score; its true error is the
 * noise-free surface.
 */
export function GridSearchSurface() {
  const state = useFigureState({
    points: int(5, { ge: 2, le: 21, suggestions: [3, 5, 6, 8, 11], label: 'values per axis r' }),
    refine: toggle(false, 'refine around the winner'),
    noise: slider(0, 0.02, 0, { step: 0.001, label: 'evaluation noise (sd)' }),
    seed: int(1, { ge: 1, le: 999, label: 'noise seed' }),
  })
  const { points, refine, noise, seed } = state
  const raster = useMemo(() => svmRaster(81), [])

  const search = useMemo(() => {
    const s = stream(`grid-search-surface/${seed}`)
    const noisy = svmTuningSurface({ noise, stream: s })
    const score = (p: Point, k: number) => noisy.value(tensor([p.logC, p.logGamma]), child(s, 'eval', k))
    const pick = (cands: Point[], offset: number) => {
      const scores = cands.map((p, k) => score(p, offset + k))
      let best = 0
      scores.forEach((v, k) => {
        if (v < scores[best]) best = k
      })
      return { cands, scores, best }
    }
    const coarse = pick(grid(SVM_SPACE, { points }), 0)
    if (!refine) return { coarse, fine: null, winner: coarse.cands[coarse.best], reported: coarse.scores[coarse.best] }
    // One coarse spacing on each side of the coarse winner, clipped to the box.
    const w = coarse.cands[coarse.best]
    const hc = (SVM_HI[0] - SVM_LO[0]) / (points - 1)
    const hg = (SVM_HI[1] - SVM_LO[1]) / (points - 1)
    const box = space({
      logC: real(Math.max(SVM_LO[0], w.logC - hc), Math.min(SVM_HI[0], w.logC + hc)),
      logGamma: real(Math.max(SVM_LO[1], w.logGamma - hg), Math.min(SVM_HI[1], w.logGamma + hg)),
    })
    const fine = pick(grid(box, { points }), coarse.cands.length)
    const fineWins = fine.scores[fine.best] < coarse.scores[coarse.best]
    return {
      coarse,
      fine,
      winner: fineWins ? fine.cands[fine.best] : w,
      reported: fineWins ? fine.scores[fine.best] : coarse.scores[coarse.best],
    }
  }, [points, refine, noise, seed])

  const trials = search.coarse.cands.length + (search.fine ? search.fine.cands.length : 0)
  const truth = svmError(search.winner.logC, search.winner.logGamma)
  const xAxis = useAxis({ label: 'log₁₀ C', range: [SVM_LO[0], SVM_HI[0]] })
  const yAxis = useAxis({ label: 'log₁₀ γ', range: [SVM_LO[1], SVM_HI[1]] })
  return (
    <Figure
      title="Grid search on an SVM's cross-validated error"
      state={state}
      caption="The colour is the cross-validated error of an RBF support vector machine on two iris species, over log C and log γ; it is clipped at 0.15 so that the diagonal valley of good settings shows. Orange points: a grid with r values per axis. Turn on the refinement to add a second grid, in green, spanning one coarse spacing on each side of the coarse winner. The winner is marked in the text colour. Change r to see that a finer grid is not always better: r = 6 lands on whole decades and hits the optimum, r = 8 misses the valley. Add evaluation noise to see the reported best score fall below the true error of the configuration it picked."
      readouts={
        <>
          <Readout label="trials" value={String(trials)} />
          <Readout label="reported best score" value={formatNumber(search.reported)} />
          <Readout label="true error of the winner" value={formatNumber(truth)} />
          <Readout label="lowest error in the table" value={formatNumber(SVM_BEST)} />
          <Readout
            label="winner (log C, log γ)"
            value={`(${formatNumber(search.winner.logC)}, ${formatNumber(search.winner.logGamma)})`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Raster
          x={raster.cAxis}
          y={raster.gammaAxis}
          z={raster.z}
          range={[SVM_BEST, 0.15]}
          valueLabel="cross-validated error"
        />
        <Points
          name="coarse grid"
          x={search.coarse.cands.map((p) => p.logC)}
          y={search.coarse.cands.map((p) => p.logGamma)}
          slot={1}
        />
        {search.fine && (
          <Points
            name="fine grid"
            x={search.fine.cands.map((p) => p.logC)}
            y={search.fine.cands.map((p) => p.logGamma)}
            slot={2}
          />
        )}
        <Points name="winner" x={[search.winner.logC]} y={[search.winner.logGamma]} emphasis />
      </Plot>
    </Figure>
  )
}
