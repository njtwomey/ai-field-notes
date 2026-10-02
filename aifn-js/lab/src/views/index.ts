/** Generic views of aifn objects. Each module appends its own exports here. */
export { formatValue } from './format'
export { StateTree } from './StateTree'
export { TracePanel, type TracePanelProps } from './TraceView'
export { type ReferenceMoments, SamplesPanel, type SamplesPanelProps } from './SamplesView'
export {
  TensorMeta,
  TensorPanel,
  type TensorPanelProps,
  TensorModePanel,
  type TensorModePanelProps,
} from './TensorView'
export {
  kindOf,
  registeredViews,
  registerKind,
  registerView,
  viewFor,
  type ViewContext,
  type ViewDef,
} from './registry'
export { QuickFigure, Show, type QuickFigureProps } from './Show'
export { quickFigure } from './quick-figure'
export { MatrixDecompositionPanel, type MatrixDecompositionPanelProps } from './MatrixDecompositionView'
export { ComputationGraphPanel, type ComputationGraphPanelProps } from './ComputationGraphView'
export { ProgramView, type ProgramViewProps } from './ProgramView'
export { TableauView, type TableauViewProps } from './TableauView'
export { DistributionPanel, type DistributionPanelProps } from './DistributionView'
export { distributionRange } from './distribution-range'
export { CurveChart, type CurveChartProps, CurvePanel, type CurvePanelProps } from './CurveView'
export { FitPanel, type FitPanelProps } from './FitView'
export { CrossValidationPanel, type CrossValidationPanelProps } from './CrossValidationView'
export { GraphView, type GraphViewProps, type PerItem } from './GraphView'
export { TreeView, type NodeSummary, type TreeViewProps } from './TreeView'
export { PlateView, type PlateViewProps } from './PlateView'
export { FactorGraphView, type FactorGraphViewProps } from './FactorGraphView'
export { ChainPanel, type ChainPanelProps } from './ChainView'
export { ElboPanel, type ElboPanelProps } from './ElboView'
export { ParamsPanel, type ParamsPanelProps } from './ParamsView'
export { DecisionRegionPanel, type DecisionRegionPanelProps } from './DecisionRegionView'
export { DendrogramPanel, type DendrogramPanelProps } from './DendrogramView'
export { DatasetPanel, type DatasetPanelProps } from './DatasetView'
export { useDatasetSeries } from './dataset-series'
export { ContingencyTableView, type ContingencyTableViewProps } from './ContingencyTableView'
export { useClassTable, type ClassTable, type ClassTableInput } from './class-table'
export { ClassLegend } from './ClassLegend'
export { PairPlotPanel, type PairPlotPanelProps } from './PairPlotView'
export { ParallelCoordinatesPanel, type ParallelCoordinatesPanelProps } from './ParallelCoordinatesView'
export { AndrewsCurvesPanel, type AndrewsCurvesPanelProps } from './AndrewsCurvesView'
export { histogramBars, type HistogramBars } from './histogram'
export { GymTrainer, type GymSetup, type GymTrainerProps } from './gym/GymTrainer'
export { gymEnvironment, gymSetup } from './gym/setup'
export { trainingRun, trainingRunOf, type TrainingRunDefaults, type TrainingRunValues } from './gym/trainingRun'
export { GYM_RENDERERS, renderKind } from './gym/registry'
export { GridView, type GridValueField, type GridViewProps } from './gym/GridView'
export { cellXY, GRID_KINDS, kindRows, LANE_OFFSET, pathMoves, policyArrows, valueRows } from './gym/grid'
export { type GymRenderer, type GymRenderProps } from './gym/renderers'
export { GYM_SERIES, actionSeries, type ActionSeries, type StateSeries } from './gym/series'
export { HypothesisTestPanel, type HypothesisTestPanelProps } from './HypothesisTestView'
export { AttentionPanel, attentionPattern, type AttentionPanelProps, type AttentionPattern } from './AttentionView'
export { TrainControls, type TrainControlsProps } from './TrainControls'
export { useTrainedRun, type TrainedRun } from './useTrainedRun'
export {
  CrfTrainingPlots,
  CrfWeightPlots,
  FiringFeatureList,
  TemplateCells,
  TemplateEditor,
  type ParsedTemplates,
} from './crf/CrfViews'
export { crfOptimiserField, crfTrainingOptions, describeOptimiser } from './crf/optimiser'
export {
  CLASSIFICATION_CASES,
  datasetChoice,
  REGRESSION_CASES,
  type DatasetCase,
  type DatasetChoice,
  type DatasetValue,
} from './dataset-choice'
export { optimiserField, optimiserLabel, trainingMethodOf, type OptimiserFieldOptions } from './optimiser-field'
