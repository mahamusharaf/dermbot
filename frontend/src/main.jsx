import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AppGate from './AppGate.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AppGate />
  </StrictMode>,
)