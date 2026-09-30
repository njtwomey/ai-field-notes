import { iris } from 'aifn-applied/data/real'
import { PairPlotView, ParallelCoordinatesView } from '@lab/views'

const IRIS = iris()

/** The multi-feature data views on Iris: a scatter matrix and parallel coordinates. */
export function DataViewsFigures() {
  return (
    <>
      <PairPlotView
        data={IRIS}
        title="PairPlotView: Iris"
        description="Four features, 16 panels: per-class histograms on the diagonal, scatters below, correlations above; x shared down columns, y across rows."
        defaultSize="M"
      />
      <ParallelCoordinatesView
        data={IRIS}
        title="ParallelCoordinatesView: Iris"
        description="One axis per feature and one line per flower; brush an axis, drag a chip to reorder, flip an axis, show the class medians."
        medians
        defaultSize="M"
      />
    </>
  )
}
