/**
 * `make lab-check`: render every lab specimen, and the lab's own pages, to HTML on the server (through Vite, so aliases
 * and TSX work) and report any that throw. Charts render their containers only (ECharts draws in the browser), so this
 * catches import errors, bad props and exceptions in specimen code, not visual problems.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { createElement, type ReactNode } from 'react'
import { renderToString } from 'react-dom/server'
import { createServer } from 'vite'

/**
 * The lab's import boundary: it depends on `aifn/<module>`, third-party packages and itself only. Fails on any import
 * under src of the site's `@/…` paths, of `site/` or `content/`, or of a relative path that leaves aifn-lab. Imports of
 * src/legacy are warnings while that folder exists.
 */
function checkImports(): number {
  const root = import.meta.dirname
  const src = path.join(root, 'src')
  const legacy = path.join(src, 'legacy')
  const renderRoot = path.resolve(root, '..', '..', 'render')
  const files = (readdirSync(src, { recursive: true }) as string[])
    .filter((f) => /\.(ts|tsx|js|jsx|mts|css)$/.test(f))
    .map((f) => path.join(src, f))
  const pattern = /(?:\bfrom\s*|\bimport\s*\(?\s*|@import\s+|@source\s+)['"]([^'"]+)['"]/g
  let errors = 0
  for (const file of files) {
    const text = readFileSync(file, 'utf8')
    for (const match of text.matchAll(pattern)) {
      const spec = match[1]
      const line = text.slice(0, match.index).split('\n').length
      const where = `${path.relative(root, file)}:${line}`
      const target = spec.startsWith('.') ? path.resolve(path.dirname(file), spec) : null
      const inLegacy = target !== null && (target === legacy || target.startsWith(legacy + path.sep))
      const inRender = target !== null && (target === renderRoot || target.startsWith(renderRoot + path.sep))
      if (inLegacy && existsSync(legacy)) {
        console.warn(`WARN  ${where}: imports src/legacy ('${spec}'); migrate to v2 (src/MIGRATION.md)`)
      } else if (inRender) {
        // Allowed: lab depends on workspace package aifn-render
      } else if (/^(@\/|site\/|content\/)/.test(spec)) {
        errors++
        console.error(`FAIL  ${where}: imports '${spec}'; the lab may import aifn, packages and @lab/… only`)
      } else if (target !== null && path.relative(root, target).startsWith('..')) {
        errors++
        console.error(`FAIL  ${where}: imports '${spec}', which is outside aifn-lab`)
      }
    }
  }
  return errors
}

const importErrors = checkImports()
// `--imports-only` (run by `make lint`): the boundary alone, without rendering.
if (process.argv.includes('--imports-only')) process.exit(importErrors ? 1 : 0)

// The theme provider and figures read the colour-scheme preference and saved sizes; give them inert stand-ins.
const store = new Map<string, string>()
Object.assign(globalThis, {
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  localStorage: {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => store.set(k, v),
    removeItem: (k: string) => store.delete(k),
  },
})

const server = await createServer({
  configFile: path.join(import.meta.dirname, 'vite.config.ts'),
  server: { middlewareMode: true, hmr: false },
  appType: 'custom',
  logLevel: 'error',
})
let failures = 0
let total = 0
try {
  const { Providers } = await server.ssrLoadModule('/src/layout/Providers.tsx')
  const { FigurePage } = await server.ssrLoadModule('/src/layout/FigurePage.tsx')
  const { specimenPath } = await server.ssrLoadModule('/src/layout/paths.ts')
  // Each page renders as the shell renders it, then its figures must have unique anchors (`#<figure-id>`).
  const render = (name: string, scope: string, node: () => ReactNode) => {
    total++
    try {
      const html = renderToString(
        createElement(Providers, { theme: 'light' }, createElement(FigurePage, { scope }, node())),
      )
      const ids = [...html.matchAll(/data-figure-id="([^"]+)"/g)].map((m) => m[1])
      const repeated = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))]
      if (repeated.length) {
        failures++
        console.error(`FAIL  ${name}: figure ids repeat within the page: ${repeated.join(', ')}`)
      }
      // Every figure states its purpose (DESIGN.md §2.1).
      const purposeless = [...html.matchAll(/data-figure-id="([^"]+)"[^>]*data-figure-purpose="missing"/g)].map(
        (m) => m[1],
      )
      if (purposeless.length) {
        failures++
        console.error(`FAIL  ${name}: figures without a purpose: ${purposeless.join(', ')}`)
      }
      // A view never renders a Figure (design S §4.1): no figure inside another's chart area.
      const nested = [...html.matchAll(/data-figure-id="([^"]+)"[^>]*data-figure-nested=""/g)].map((m) => m[1])
      if (nested.length) {
        failures++
        console.error(`FAIL  ${name}: figures inside another figure: ${nested.join(', ')}`)
      }
    } catch (e) {
      failures++
      console.error(`FAIL  ${name}: ${(e as Error).message.split('\n')[0]}`)
    }
  }
  const paths = new Map<string, string>()
  const modules = new Set<string>()
  const { UiKit } = await server.ssrLoadModule('/src/kit/UiKit.tsx')
  render('lab / UI kit', 'ui-kit', () => createElement(UiKit))
  const { plotKitPages } = await server.ssrLoadModule('/src/kit/plot/pages.tsx')
  for (const p of plotKitPages as { key: string; title: string; render: () => ReactNode }[])
    render(`lab / ${p.title}`, p.key, p.render)
  const { stateKitPages } = await server.ssrLoadModule('/src/kit/state/pages.tsx')
  for (const p of stateKitPages as { key: string; title: string; render: () => ReactNode }[])
    render(`lab / ${p.title}`, p.key, p.render)
  const { DiagramsKit } = await server.ssrLoadModule('/src/kit/DiagramsKit.tsx')
  render('lab / Diagrams', 'diagrams', () => createElement(DiagramsKit))
  // Specimen files mirror the module tree (`<family>/<module>.tsx`); `_`-folders and `_`-files hold shared code.
  const files = (readdirSync(path.join(import.meta.dirname, 'src/specimens'), { recursive: true }) as string[])
    .map((f) => f.split(path.sep).join('/'))
    .filter((f) => f.endsWith('.tsx') && !f.split('/').some((part) => part.startsWith('_')))
  for (const file of files.sort()) {
    const mod = await server.ssrLoadModule(`/src/specimens/${file}`)
    for (const s of mod.specimens ?? []) {
      const at = specimenPath(s) as string
      if (paths.has(at)) {
        failures++
        console.error(`FAIL  ${s.module} / ${s.title}: path /${at} is also ${paths.get(at)}'s`)
      }
      paths.set(at, `${s.module} / ${s.title}`)
      modules.add(s.module)
      render(`${s.module} / ${s.title}`, at, () => s.render())
    }
  }
  // The sidebar files each module under its family; one missing from the map lands under "Other".
  const { unmappedModules } = await server.ssrLoadModule('/src/layout/families.ts')
  for (const m of unmappedModules(modules) as string[])
    console.warn(`WARN  ${m}: module missing from src/layout/families.ts; the sidebar lists it under Other`)
} finally {
  await server.close()
}
console.log(`${total} pages and specimens · ${failures} failed · ${importErrors} import boundary violations`)
if (failures || importErrors) process.exit(1)
