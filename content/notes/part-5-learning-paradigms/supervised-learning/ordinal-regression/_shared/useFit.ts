import { useEffect, useState } from 'react'
import {
  cachedFit,
  dataset,
  fitModel,
  specKey,
  startFit,
  type DataSpec,
  type Dataset,
  type Fitted,
  type ModelId,
} from './ordinal'

/** Milliseconds of optimisation per slice before yielding to the browser. */
const SLICE_MS = 25

/**
 * A model fitted to a spec. Cached fits come back at once. A new fit runs to convergence in short slices between
 * frames, starting from the last solution for the same curve, K and model; until it converges the previous fit stays on
 * screen with its own dataset, and `fitting` is true, so sliders and drags stay responsive however long the fit takes.
 */
export function useFit(spec: DataSpec, model: ModelId): { fitted: Fitted; data: Dataset; fitting: boolean } {
  const key = `${specKey(spec)}/${model}`
  // The first render fits synchronously, so there is always something to draw; the defaults fit in well under 0.2 s.
  const [done, setDone] = useState<{ key: string; spec: DataSpec; fitted: Fitted }>(() => ({
    key,
    spec,
    fitted: fitModel(spec, model),
  }))
  const cached = cachedFit(spec, model)

  useEffect(() => {
    if (cachedFit(spec, model)) return
    const job = startFit(spec, model)
    let timer = 0
    const tick = () => {
      const fitted = job.step(SLICE_MS)
      if (fitted) setDone({ key, spec, fitted })
      else timer = window.setTimeout(tick, 0)
    }
    timer = window.setTimeout(tick, 0)
    return () => window.clearTimeout(timer)
    // The key captures every field of the spec and the model.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  // While a fit runs, the data shown is the data the previous fit was made on, so K and the points always match it.
  if (cached) return { fitted: cached, data: dataset(spec), fitting: false }
  return { fitted: done.fitted, data: dataset(done.spec), fitting: done.key !== key }
}
