/** The lab's controls. Everything outside src/controls imports from '@lab/controls' only. */
export { Slider, type SliderProps } from './Slider'
export { NumberField, type NumberFieldProps } from './NumberField'
export { Select, type SelectProps } from './Select'
export { Combobox, type ComboboxProps } from './Combobox'
export { MultiCombobox, type MultiComboboxProps } from './MultiCombobox'
export { Choice, SEARCHABLE_FROM, type ChoiceProps } from './Choice'
export { Switch } from './Switch'
export { Player, type PlayerProps } from './Player'
export { usePlayhead } from './playhead'
export { StepControls } from './StepControls'
export { ControlLabel } from './ControlLabel'
export { useParam, type Param, type ParamSpec } from './param'
export {
  slider,
  number,
  float,
  int,
  choice,
  toggle,
  setting,
  row,
  variants,
  when,
  type ParamDef,
  type ParamDefs,
  type ParamValue,
  type Values,
  type ValueOf,
} from './params'
export { FigureControls, LeafControl, ParamControls, type ParamControlsProps } from './ParamControls'
export {
  defineVariants,
  useVariants,
  useParams,
  type Variants,
  type VariantsControl,
  type VariantsJson,
} from './variants'
export { VariantControls } from './VariantControls'
export type { Option, Options } from './options'
// Buttons come straight from the shadcn primitives; re-exported so figures need one import for their controls.
export { Button } from '@lab/ui/button'
export { ButtonGroup, ButtonGroupSeparator, ButtonGroupText } from '@lab/ui/button-group'
export { CodeEditor, type CodeEditorProps, type CodeError } from './CodeEditor'
export { StatusText } from './StatusText'
