import {
  ClippedVectorFigure,
  ClusterFigure,
  DerivativeFigure,
  HistogramFigure,
  LogFigure,
  VectorFigure,
} from './ChartDemos'
import { ControlsDemo } from './ControlsDemo'
import { NumberFieldDemo } from './NumberFieldDemo'
import { FunctionFamilyFigure } from './FunctionFamily'
import { HeatmapFigure, LinkedFigure } from './HeatmapDemos'
import { PaletteDemo } from './PaletteDemo'
import { Section } from './Section'
import { GridFigure, SharedXFigure, SharedYFigure } from './SubplotsDemos'
import { DataViewsFigures } from './DataViewsDemos'

/**
 * The UI kit: a review page for the v2 building blocks. Every control, chart type, zoom and pan, hover, handles, figure
 * sizing, data export and both themes appear here, against seeded data from aifn.
 */
export function UiKit() {
  return (
    <div className="flex flex-col gap-10">
      <Section
        title="Theme and palette"
        description="The lab owns its tokens: shadcn colours in lab.css, data colours in src/design/palette.ts (seeded from the site's palette). Switch the theme here or in the sidebar."
      >
        <PaletteDemo />
      </Section>
      <Section
        title="Controls"
        description="Sliders are steppable by default and take a typed value; categorical choices are dropdowns, searchable when long."
      >
        <ControlsDemo />
      </Section>
      <Section
        title="Typed numbers"
        description="NumberField, from number(), float() and int(): a type, strict or inclusive bounds (gt, ge, lt, le), a linear or log10 scale and optional suggestions. A draft that breaks a rule is refused with a message, not clamped; the buttons step and clamp."
      >
        <NumberFieldDemo />
      </Section>
      <Section
        title="Conditional controls"
        description="defineVariants: each function brings its own parameters; the controls follow the chosen one, and each keeps its values. Params can also show conditionally within a variant."
      >
        <FunctionFamilyFigure />
      </Section>
      <Section
        title="Hover"
        description="Line charts hover by x: every series' value at the pointer, in the tooltip and in the figure's readout."
      >
        <DerivativeFigure />
      </Section>
      <Section
        title="Charts, handles and zoom"
        description="Every figure frame has a size picker (S, M, L, XL, full width), a corner to drag, and a button that copies its data as JSON. Sizes are remembered per page."
      >
        <ClusterFigure />
        <HistogramFigure />
        <LogFigure />
        <VectorFigure />
        <ClippedVectorFigure />
      </Section>
      <Section
        title="Rasters"
        description="A Raster layer (sequential, diverging or categorical) under Contours, a live path, Vectors and a Handle, with zoom and pan."
      >
        <HeatmapFigure />
      </Section>
      <Section
        title="Plots grids"
        description="Plots: aligned plot areas, axes shared by passing one axis model to several Plots (one zoom for all), tick labels on the outer panels only, height and width ratios within the frame."
      >
        <SharedXFigure />
        <SharedYFigure />
        <GridFigure />
      </Section>
      <Section
        title="Data views"
        description="PairPlotPanel and ParallelCoordinatesPanel from @lab/views: a labelled dataset in every feature at once, classes in their categorical slots, class chips that hide a class everywhere, and brushing that filters rows across panels or axes."
      >
        <DataViewsFigures />
      </Section>
      <Section
        title="Linked panels"
        description="Plots in one grid share the hovered x (hoverGroup) and one step axis; the grid splits the frame among them."
      >
        <LinkedFigure />
      </Section>
    </div>
  )
}
