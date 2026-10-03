import { lazy, Suspense } from 'react'
import { createBrowserRouter, RouterProvider } from 'react-router'
import { registerMathMacros, RenderMathProvider } from 'aifn-render'
import { macros } from '@content/macros'
import { ThemeProvider } from '@/components/theme-provider'
import { AppShell } from '@/components/layout/AppShell'
import { TooltipProvider } from '@/components/ui/tooltip'
import { BrowsePage, CategoryRedirect } from '@/pages/BrowsePage'
import { DiagramLabPage } from '@/pages/DiagramLabPage'
import { GlossaryPage } from '@/pages/GlossaryPage'
import { HomePage } from '@/pages/HomePage'
import { NotePage } from '@/pages/NotePage'
import { NotFoundPage } from '@/pages/NotFoundPage'
import { ReferencesPage } from '@/pages/ReferencesPage'
import { TagPage, TagsPage } from '@/pages/TagsPage'

import { RouteErrorBoundary } from '@/components/layout/RouteErrorBoundary'

// Register site macros globally for aifn-render components
registerMathMacros(macros)

// Loads KaTeX's renderer, so it gets its own chunk.
const NotationPage = lazy(() => import('@/pages/NotationPage'))

const router = createBrowserRouter(
  [
    {
      element: <AppShell />,
      errorElement: <RouteErrorBoundary />,
      children: [
        { index: true, element: <HomePage /> },
        { path: 'browse', element: <BrowsePage /> },
        { path: 'c/*', element: <CategoryRedirect /> },
        { path: 'tags', element: <TagsPage /> },
        { path: 'tags/:tag', element: <TagPage /> },
        { path: 'references', element: <ReferencesPage /> },
        { path: 'glossary', element: <GlossaryPage /> },
        { path: 'lab/diagrams', element: <DiagramLabPage /> },
        {
          path: 'notation',
          element: (
            <Suspense>
              <NotationPage />
            </Suspense>
          ),
        },
        { path: 'n/:slug', element: <NotePage tab="concept" /> },
        { path: 'n/:slug/code', element: <NotePage tab="code" /> },
        { path: 'n/:slug/outputs/:run?', element: <NotePage tab="outputs" /> },
        { path: '*', element: <NotFoundPage /> },
      ],
    },
  ],
  { basename: import.meta.env.BASE_URL.replace(/\/$/, '') },
)

export default function App() {
  return (
    <ThemeProvider>
      <RenderMathProvider macros={macros}>
        <TooltipProvider>
          <RouterProvider router={router} />
        </TooltipProvider>
      </RenderMathProvider>
    </ThemeProvider>
  )
}

