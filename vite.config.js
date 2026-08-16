import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import capture from './vite-plugin-capture.js';
import figures from './vite-plugin-figures.js';

export default defineConfig({
  plugins: [react(), capture(), figures()],
  server: { port: 5183 },
});
