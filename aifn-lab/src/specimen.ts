import type { ReactNode } from 'react'

/**
 * A specimen: one small, self-contained use of an aifn module. It declares what it runs and returns something the lab
 * can show. Modules ship several; the lab lists and searches them, and the test suite runs them as smoke tests.
 */
export type Specimen = {
  /** The aifn module it exercises, e.g. `special`, `linalg`. */
  module: string
  title: string
  description: string
  tags?: string[]
  /** Renders the specimen; views from `@lab/views` show aifn objects generically. */
  render: () => ReactNode
}
