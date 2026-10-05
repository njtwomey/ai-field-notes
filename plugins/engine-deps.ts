import fs from 'node:fs'
import path from 'node:path'

const ENGINE = ['aifn-compute', 'aifn-methods', 'aifn-render'] as const

/**
 * Every JavaScript entry of the installed engine packages (`aifn-compute/optim`, `aifn-render/viz`, …), from their
 * `exports`, for Vite's `optimizeDeps.include`.
 *
 * Pre-bundling them all at start-up does two things. Vite bundles them in one run with shared chunks, so compute's
 * primitive registry is one module however many entries a page imports (a second registration throws). And no page
 * discovers a new entry later, which would make Vite re-optimise and reload the page mid-load. Stylesheets and the
 * compute worker are left out: the worker is bundled from its `new URL(…, import.meta.url)`.
 */
export function engineDeps(root: string): string[] {
  return ENGINE.flatMap((pkg) => {
    const manifest = path.join(root, 'node_modules', pkg, 'package.json')
    const exports = (JSON.parse(fs.readFileSync(manifest, 'utf8')) as { exports: Record<string, unknown> }).exports
    return Object.keys(exports)
      .filter((k) => !k.includes('*') && !k.endsWith('.css') && !k.endsWith('.worker'))
      .map((k) => (k === '.' ? pkg : `${pkg}/${k.slice(2)}`))
  })
}
