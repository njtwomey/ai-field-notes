import { iris } from 'aifn-methods/data/real'
import { Figure } from '@lab/layout'
import { PairPlotPanel, ParallelCoordinatesPanel } from '@lab/views'

const IRIS = iris()

/** The multi-feature data views on Iris: a scatter matrix and parallel coordinates, each a panel in a Figure. */
export function DataViewsFigures() {
  return (
    <>
      <Figure
        title="PairPlotPanel: Iris"
        purpose="Four features, 16 panels: per-class histograms on the diagonal, scatters below, correlations above; x shared down columns, y across rows."
        caption="Drag a rectangle in any scatter to select the rows inside it; they keep their colour in every panel and the rest fade. A click without a drag clears it. Hover a point to mark the same row everywhere."
        hoverReadout={false}
        defaultSize="M"
      >
        <PairPlotPanel data={IRIS} />
      </Figure>
      <Figure
        title="ParallelCoordinatesPanel: Iris"
        purpose="One axis per feature and one line per flower; brush an axis, drag a chip to reorder, flip an axis, show the class medians."
        caption="Drag along an axis to brush an interval (drag again to add one; click the axis outside it to clear). Drag a chip in the header to move its axis; its arrow flips the axis."
        hoverReadout={false}
        defaultSize="M"
      >
        <ParallelCoordinatesPanel data={IRIS} medians />
      </Figure>
    </>
  )
}
