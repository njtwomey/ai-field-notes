import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Providers } from 'aifn-render/layout'
import { App } from './App'
import './lab.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Providers>
      <App />
    </Providers>
  </StrictMode>,
)
