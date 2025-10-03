import React from 'react'
import './index.css'
import { createRoot } from 'react-dom/client'
import CreditScoringApp from './creditScoringApp'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <CreditScoringApp />
  </React.StrictMode>
)
