/**
 * Jest Configuration for Hospital-Grade Medication System
 * Comprehensive testing setup with coverage requirements
 */

module.exports = {
  // Test environment
  testEnvironment: 'node',

  // Test directories
  testMatch: [
    '**/tests/unit/**/*.test.js',
    '**/tests/integration/**/*.test.js'
  ],

  // Setup files
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],

  // Coverage configuration - Hospital Grade Requirements: ≥95%
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html', 'json'],
  
  coverageThreshold: {
    global: {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    // Specific thresholds for critical components
    './src/services/ProfoundMedicationResolver.js': {
      branches: 98,
      functions: 98,
      lines: 98,
      statements: 98
    },
    './src/controllers/HospitalGradeMedicationController.js': {
      branches: 98,
      functions: 98,
      lines: 98,
      statements: 98
    },
    './src/controllers/medicationController.js': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },

  // Files to collect coverage from
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
    '!src/**/*.spec.js',
    '!src/tests/**',
    '!src/**/node_modules/**',
    '!src/config/database.js', // Exclude database config
    '!src/app.js' // Exclude main app file
  ],

  // Module directories
  moduleDirectories: ['node_modules', 'src'],

  // Test timeout (important for database operations)
  testTimeout: 30000,

  // Globals
  globals: {
    'process.env.NODE_ENV': 'test',
    'process.env.DATABASE_URL': 'file:./test.db'
  },

  // Transform configuration
  transform: {
    '^.+\\.js$': 'babel-jest'
  },

  // Module name mapping
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1'
  },

  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/build/',
    '/dist/',
    '/coverage/'
  ],

  // Clear mocks between tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,

  // Verbose output for debugging
  verbose: true,

  // Bail on first test suite failure in CI
  bail: process.env.CI ? 1 : 0,

  // Max workers for parallel testing
  maxWorkers: process.env.CI ? 2 : '50%',

  // Error on deprecated features
  errorOnDeprecated: true,

  // Notify mode (only in development)
  notify: !process.env.CI,
  notifyMode: 'failure-change',

  // Custom reporters
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'coverage',
      outputName: 'junit.xml',
      usePathForSuiteName: true
    }]
  ],

  // Additional setup for hospital-grade requirements
  setupFiles: ['<rootDir>/tests/hospital-grade-setup.js']
};
