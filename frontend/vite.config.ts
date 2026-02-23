import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

const proxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/health': proxyTarget,
      '/ingest': proxyTarget,
      '/features': proxyTarget,
      '/train': proxyTarget,
      '/runs': proxyTarget,
      '/predict': proxyTarget,
    },
  }
});
