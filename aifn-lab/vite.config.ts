import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const src = path.join(import.meta.dirname, 'src')
const modulesJson = path.join(import.meta.dirname, '..', 'aifn-js', 'modules.json')

/**
 * The aifn lab: a standalone app for exploring aifn. It depends on aifn and aifn-applied (the workspace packages),
 * `aifn-js/modules.json` (the module order, for the sidebar) and third-party packages only; nothing is imported from
 * the site.
 */
export default defineConfig({
  root: import.meta.dirname,
  plugins: [
    react(),
    tailwindcss(),
    // aifn registers each primitive once at import, and a second registration throws; so an edit to aifn-js reloads
    // the page (a fresh registry) instead of re-running the edited module under hot update. Any file under aifn-js
    // counts, at any depth of the module tree.
    {
      name: 'aifn-full-reload',
      handleHotUpdate({ file, server }) {
        if (!file.includes(`${path.sep}aifn-js${path.sep}`)) return
        server.ws.send({ type: 'full-reload' })
        return []
      },
    },
  ],
  resolve: {
    alias: [
      { find: /^@lab\//, replacement: `${src}/` },
      { find: /^aifn-js\/modules\.json$/, replacement: modulesJson },
    ],
  },
  // A single-page app with path URLs (/<module>/<specimen>): dev and preview serve index.html for any unknown path.
  appType: 'spa',
  server: { port: 5190 },
  preview: { port: 5190 },
})
