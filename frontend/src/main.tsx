import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { installWheelScrollAssist } from './lib/scrollAssist'
import PlatformRouter from './platform/PlatformRouter'
import PrettyTooltipLayer from './shared/PrettyTooltipLayer'
import './index.css'
import './workbench-layout.css'

installWheelScrollAssist()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <PlatformRouter />
    </BrowserRouter>
    <PrettyTooltipLayer />
  </React.StrictMode>,
)
