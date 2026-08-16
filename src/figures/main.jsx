import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import FiguresApp from './FiguresApp.jsx';
import './figures.css';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <FiguresApp />
  </StrictMode>,
);
