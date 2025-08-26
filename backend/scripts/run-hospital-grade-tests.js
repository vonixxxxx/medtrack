#!/usr/bin/env node

/**
 * Hospital-Grade Test Runner
 * Enforces zero tolerance for failures and ≥95% coverage requirements
 */

const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class HospitalGradeTestRunner {
  constructor() {
    this.results = {
      unit: null,
      integration: null,
      e2e: null,
      coverage: null
    };
    this.startTime = Date.now();
  }

  async run() {
    console.log('🏥 Starting Hospital-Grade Test Suite');
    console.log('   Zero tolerance for failures');
    console.log('   ≥95% coverage required');
    console.log('   All browsers must pass\n');

    try {
      // 1. Unit Tests
      await this.runUnitTests();
      
      // 2. Integration Tests  
      await this.runIntegrationTests();
      
      // 3. Coverage Verification
      await this.verifyCoverage();
      
      // 4. E2E Tests
      await this.runE2ETests();
      
      // 5. Final Validation
      await this.validateHospitalGradeRequirements();
      
      console.log('\n✅ HOSPITAL-GRADE TEST SUITE PASSED');
      process.exit(0);
      
    } catch (error) {
      console.error('\n❌ HOSPITAL-GRADE TEST SUITE FAILED');
      console.error(`   ${error.message}`);
      process.exit(1);
    }
  }

  async runUnitTests() {
    console.log('🧪 Running Unit Tests...');
    
    try {
      await this.runCommand('npx', ['jest', '--config=jest.config.js', '--testPathPattern=unit', '--coverage']);
      
      const coverage = await this.parseCoverageResults();
      this.results.unit = { passed: true, coverage };
      
      console.log('✅ Unit Tests Passed');
      
    } catch (error) {
      throw new Error(`Unit tests failed: ${error.message}`);
    }
  }

  async runIntegrationTests() {
    console.log('🔗 Running Integration Tests...');
    
    try {
      await this.runCommand('npx', ['jest', '--config=jest.config.js', '--testPathPattern=integration']);
      
      this.results.integration = { passed: true };
      console.log('✅ Integration Tests Passed');
      
    } catch (error) {
      throw new Error(`Integration tests failed: ${error.message}`);
    }
  }

  async verifyCoverage() {
    console.log('📊 Verifying Coverage Requirements...');
    
    try {
      const coveragePath = path.join(__dirname, '../coverage/coverage-summary.json');
      const coverageData = JSON.parse(await fs.readFile(coveragePath, 'utf8'));
      
      const total = coverageData.total;
      const requirements = {
        lines: total.lines.pct >= 95,
        functions: total.functions.pct >= 95, 
        branches: total.branches.pct >= 95,
        statements: total.statements.pct >= 95
      };
      
      console.log(`   Lines: ${total.lines.pct}% ${requirements.lines ? '✅' : '❌'}`);
      console.log(`   Functions: ${total.functions.pct}% ${requirements.functions ? '✅' : '❌'}`);
      console.log(`   Branches: ${total.branches.pct}% ${requirements.branches ? '✅' : '❌'}`);
      console.log(`   Statements: ${total.statements.pct}% ${requirements.statements ? '✅' : '❌'}`);
      
      const allRequirementsMet = Object.values(requirements).every(req => req === true);
      
      if (!allRequirementsMet) {
        throw new Error('Coverage requirements not met (≥95% required for all metrics)');
      }
      
      this.results.coverage = { passed: true, metrics: total };
      console.log('✅ Coverage Requirements Met');
      
    } catch (error) {
      throw new Error(`Coverage verification failed: ${error.message}`);
    }
  }

  async runE2ETests() {
    console.log('🎭 Running E2E Tests...');
    
    try {
      await this.runCommand('npx', ['playwright', 'test']);
      
      const e2eResults = await this.parseE2EResults();
      this.results.e2e = e2eResults;
      
      if (e2eResults.failed > 0) {
        throw new Error(`E2E tests failed: ${e2eResults.failed} failures`);
      }
      
      console.log('✅ E2E Tests Passed');
      
    } catch (error) {
      throw new Error(`E2E tests failed: ${error.message}`);
    }
  }

  async validateHospitalGradeRequirements() {
    console.log('🏥 Validating Hospital-Grade Requirements...');
    
    const requirements = {
      unitTestsPassed: this.results.unit?.passed === true,
      integrationTestsPassed: this.results.integration?.passed === true,
      coverageAbove95: this.results.coverage?.passed === true,
      e2eTestsPassed: this.results.e2e?.failed === 0,
      zeroFailures: (this.results.e2e?.failed || 0) === 0,
      allBrowsersTested: (this.results.e2e?.browsers || []).length >= 3
    };
    
    const allMet = Object.values(requirements).every(req => req === true);
    
    console.log('\n📋 Hospital-Grade Requirements Check:');
    Object.entries(requirements).forEach(([requirement, met]) => {
      console.log(`   ${requirement}: ${met ? '✅' : '❌'}`);
    });
    
    if (!allMet) {
      throw new Error('Hospital-grade requirements not met');
    }
    
    // Generate compliance report
    await this.generateComplianceReport(requirements);
    
    console.log('✅ Hospital-Grade Requirements Validated');
  }

  async runCommand(command, args) {
    return new Promise((resolve, reject) => {
      const process = spawn(command, args, {
        stdio: 'inherit',
        shell: true
      });
      
      process.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Command failed with exit code ${code}`));
        }
      });
      
      process.on('error', (error) => {
        reject(error);
      });
    });
  }

  async parseCoverageResults() {
    try {
      const coveragePath = path.join(__dirname, '../coverage/coverage-summary.json');
      const coverageData = JSON.parse(await fs.readFile(coveragePath, 'utf8'));
      return coverageData.total;
    } catch (error) {
      console.warn('⚠️ Could not parse coverage results:', error.message);
      return null;
    }
  }

  async parseE2EResults() {
    try {
      const resultsPath = path.join(__dirname, '../test-results/results.json');
      const resultsData = JSON.parse(await fs.readFile(resultsPath, 'utf8'));
      
      return {
        total: resultsData.stats?.total || 0,
        passed: resultsData.stats?.passed || 0,
        failed: resultsData.stats?.failed || 0,
        skipped: resultsData.stats?.skipped || 0,
        browsers: resultsData.config?.projects?.map(p => p.name) || []
      };
    } catch (error) {
      console.warn('⚠️ Could not parse E2E results:', error.message);
      return { total: 0, passed: 0, failed: 1, skipped: 0, browsers: [] };
    }
  }

  async generateComplianceReport(requirements) {
    const report = {
      timestamp: new Date().toISOString(),
      testSuiteDuration: Date.now() - this.startTime,
      hospitalGradeCompliant: Object.values(requirements).every(req => req === true),
      requirements,
      results: this.results,
      metadata: {
        zeroTolerancePolicy: 'Hospital-grade systems require zero tolerance for failures',
        coverageRequirement: '≥95% coverage required for all metrics',
        browserSupport: 'All major browsers must pass E2E tests',
        systemType: 'Hospital-Grade Medication Management System'
      }
    };
    
    const reportPath = path.join(__dirname, '../test-results/hospital-grade-compliance.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`📋 Compliance report generated: ${reportPath}`);
  }
}

// Script execution
if (require.main === module) {
  const runner = new HospitalGradeTestRunner();
  runner.run().catch(error => {
    console.error('💥 Test runner crashed:', error);
    process.exit(1);
  });
}

module.exports = HospitalGradeTestRunner;
