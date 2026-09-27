import { defineConfig, devices } from '@playwright/test';

// Tests E2E du parcours critique. Le serveur Streamlit est lancé par Playwright sur un port
// dédié (jamais celui du développement), en local uniquement.
const PORT = 8599;

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 240_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    locale: 'fr-FR',
    timezoneId: 'UTC',
    trace: 'retain-on-failure',
    acceptDownloads: true,
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: {
    command:
      `.venv/bin/python -m streamlit run app.py --server.headless true --server.port ${PORT} ` +
      '--server.address 127.0.0.1 --browser.gatherUsageStats false',
    url: `http://127.0.0.1:${PORT}/_stcore/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: { OMP_NUM_THREADS: '1' },
  },
});
