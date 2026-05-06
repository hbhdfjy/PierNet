import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { installWheelScrollAssist } from './lib/scrollAssist'
import PrettyTooltipLayer from './shared/PrettyTooltipLayer'
import './index.css'

installWheelScrollAssist()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
    <PrettyTooltipLayer />
  </React.StrictMode>,
)
