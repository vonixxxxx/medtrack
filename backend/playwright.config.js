/**
 * Playwright Configuration for Hospital-Grade E2E Testing
 * Zero tolerance for failures - all tests must pass
 */

const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  // Test directory
  testDir: './tests/e2e',

  // Timeout configuration
  timeout: 30 * 1000, // 30 seconds per test
  expect: {
    timeout: 5 * 1000 // 5 seconds for assertions
  },

  // Hospital-Grade Requirements: All tests must pass
  fullyParallel: true,
  forbidOnly: !!process.env.CI, // Fail if .only is left in CI
  retries: process.env.CI ? 2 : 0, // Retry failed tests in CI
  workers: process.env.CI ? 1 : undefined, // Run sequentially in CI
  
  // Reporting
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/junit.xml' }]
  ],

  // Global test configuration
  use: {
    // Base URL for tests
    baseURL: 'http://localhost:3005',

    // Browser context options
    ignoreHTTPSErrors: true,
    
    // Screenshots and videos for debugging
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',

    // Network settings
    navigationTimeout: 10 * 1000,
    actionTimeout: 5 * 1000
  },

  // Projects for different browsers - Hospital Grade Testing
  projects: [
    // Desktop browsers
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    // Mobile browsers
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },

    // Tablet
    {
      name: 'iPad',
      use: { ...devices['iPad Pro'] },
    }
  ],

  // Web server configuration
  webServer: [
    {
      command: 'cd ../frontend && npm run dev -- --port 3005',
      port: 3005,
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000
    },
    {
      command: 'npm start',
      port: 8000,
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000
    }
  ],

  // Global setup and teardown
  globalSetup: require.resolve('./tests/e2e/global-setup.js'),
  globalTeardown: require.resolve('./tests/e2e/global-teardown.js'),

  // Output directories
  outputDir: 'test-results/',

  // Hospital-Grade specific configuration
  metadata: {
    testType: 'hospital-grade-e2e',
    zeroToleranceForFailures: true,
    requiresAllBrowsers: true,
    accessibilityCompliant: true
  }
});
