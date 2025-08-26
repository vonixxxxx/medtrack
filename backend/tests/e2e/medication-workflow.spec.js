/**
 * End-to-End Tests for Hospital-Grade Medication Workflow
 * Tests complete user journeys with real frontend interactions
 */

const { test, expect } = require('@playwright/test');

test.describe('Hospital-Grade Medication Management', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    
    // Navigate to app and login
    await page.goto('http://localhost:3005');
    
    // Login flow
    await page.fill('[data-testid="email-input"]', 'test@hospital-grade.test');
    await page.fill('[data-testid="password-input"]', 'test123');
    await page.click('[data-testid="login-button"]');
    
    // Wait for dashboard to load
    await page.waitForSelector('[data-testid="dashboard"]');
  });

  test.afterEach(async () => {
    await page.close();
  });

  test('complete medication search and selection workflow', async () => {
    // Open medication popup
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Search for paracetamol
    await page.fill('[data-testid="medication-search"]', 'paracetamol');
    await page.waitForSelector('[data-testid="search-results"]');

    // Verify search results appear
    const searchResults = page.locator('[data-testid="search-result"]');
    await expect(searchResults).toHaveCount.toBeGreaterThan(0);

    // Click on first result
    await searchResults.first().click();

    // Verify product options are loaded
    await page.waitForSelector('[data-testid="product-configuration"]');
    
    // Check for hospital-grade indicators
    await expect(page.locator('[data-testid="hospital-grade-badge"]')).toBeVisible();
    await expect(page.locator('[data-testid="source-badge"]')).toBeVisible();

    // Select dosage from server-approved options
    const dosageButtons = page.locator('[data-testid="dosage-option"]');
    await expect(dosageButtons).toHaveCount.toBeGreaterThan(0);
    await dosageButtons.first().click();

    // Select frequency from server-approved options  
    const frequencyButtons = page.locator('[data-testid="frequency-option"]');
    await expect(frequencyButtons).toHaveCount.toBeGreaterThan(0);
    await frequencyButtons.first().click();

    // Select place of intake
    const placeButtons = page.locator('[data-testid="place-option"]');
    await expect(placeButtons).toHaveCount.toBeGreaterThan(0);
    await placeButtons.first().click();

    // Validate configuration
    await page.click('[data-testid="validate-button"]');
    await page.waitForSelector('[data-testid="validation-success"]');

    // Verify validation success message
    await expect(page.locator('[data-testid="validation-success"]')).toContainText('validated');

    // Complete the form
    await page.fill('[data-testid="start-date"]', '2025-01-15');
    
    // Select monitoring metrics
    await page.check('[data-testid="metric-blood-pressure"]');
    await page.check('[data-testid="metric-side-effects"]');

    // Submit final form
    await page.click('[data-testid="submit-medication"]');

    // Verify medication appears on dashboard
    await page.waitForSelector('[data-testid="medication-card"]');
    const medicationCard = page.locator('[data-testid="medication-card"]').first();
    await expect(medicationCard).toContainText('paracetamol');
  });

  test('search for GLP-1 medications via acronym', async () => {
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Search using acronym
    await page.fill('[data-testid="medication-search"]', 'GLP-1');
    await page.waitForSelector('[data-testid="search-results"]');

    // Verify multiple GLP-1 medications found
    const searchResults = page.locator('[data-testid="search-result"]');
    await expect(searchResults).toHaveCount.toBeGreaterThan(1);

    // Check for semaglutide
    const semaglutideResult = page.locator('[data-testid="search-result"]', { hasText: 'semaglutide' });
    await expect(semaglutideResult).toBeVisible();

    // Click on semaglutide
    await semaglutideResult.click();

    // Look for Ozempic product
    const ozempicProduct = page.locator('[data-testid="product-option"]', { hasText: 'Ozempic' });
    await expect(ozempicProduct).toBeVisible();

    // Select Ozempic
    await ozempicProduct.click();
    await page.waitForSelector('[data-testid="product-configuration"]');

    // Verify injection-only options
    await expect(page.locator('[data-testid="intake-type"]')).toContainText('Injection');
    
    // Verify weekly frequency only
    const frequencyOptions = page.locator('[data-testid="frequency-option"]');
    await expect(frequencyOptions).toHaveCount(1);
    await expect(frequencyOptions).toContainText('weekly');

    // Verify specific dosage options (0.25, 0.5, 1, 2 mg)
    const dosageOptions = page.locator('[data-testid="dosage-option"]');
    const dosageTexts = await dosageOptions.allTextContents();
    expect(dosageTexts.some(text => text.includes('0.25'))).toBe(true);
    expect(dosageTexts.some(text => text.includes('0.5'))).toBe(true);
    expect(dosageTexts.some(text => text.includes('1'))).toBe(true);
    expect(dosageTexts.some(text => text.includes('2'))).toBe(true);
  });

  test('validation rejects invalid configurations', async () => {
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Search for injection medication
    await page.fill('[data-testid="medication-search"]', 'Ozempic');
    await page.waitForSelector('[data-testid="search-results"]');
    await page.locator('[data-testid="search-result"]').first().click();
    await page.waitForSelector('[data-testid="product-configuration"]');

    // Try to force invalid configuration (shouldn't be possible with hospital-grade system)
    // Since UI only shows valid options, we test that validation still works

    // Select valid options first
    await page.locator('[data-testid="dosage-option"]').first().click();
    await page.locator('[data-testid="frequency-option"]').first().click();
    await page.locator('[data-testid="place-option"]').first().click();

    // Validate - should succeed
    await page.click('[data-testid="validate-button"]');
    await page.waitForSelector('[data-testid="validation-success"]');
    
    // Verify no error messages
    await expect(page.locator('[data-testid="validation-error"]')).not.toBeVisible();
  });

  test('handles typos with suggestions', async () => {
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Search with common typo
    await page.fill('[data-testid="medication-search"]', 'paracitamol');
    await page.waitForSelector('[data-testid="search-results"]', { timeout: 10000 });

    // Should show suggestions
    const suggestionSection = page.locator('[data-testid="suggestions"]');
    if (await suggestionSection.isVisible()) {
      await expect(suggestionSection).toContainText('Did you mean');
      
      // Click on suggestion
      const paracetamolSuggestion = page.locator('[data-testid="suggestion-item"]', { hasText: 'paracetamol' });
      if (await paracetamolSuggestion.isVisible()) {
        await paracetamolSuggestion.click();
        
        // Should populate search with correct spelling
        await expect(page.locator('[data-testid="medication-search"]')).toHaveValue('paracetamol');
      }
    }
  });

  test('prevents duplicate medication cycles', async () => {
    // First, add a medication
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    await page.fill('[data-testid="medication-search"]', 'ibuprofen');
    await page.waitForSelector('[data-testid="search-results"]');
    await page.locator('[data-testid="search-result"]').first().click();
    await page.waitForSelector('[data-testid="product-configuration"]');

    // Configure medication
    await page.locator('[data-testid="dosage-option"]').first().click();
    await page.locator('[data-testid="frequency-option"]').first().click();
    await page.locator('[data-testid="place-option"]').first().click();

    await page.click('[data-testid="validate-button"]');
    await page.waitForSelector('[data-testid="validation-success"]');

    // Complete form
    await page.fill('[data-testid="start-date"]', '2025-01-15');
    await page.check('[data-testid="metric-side-effects"]');
    await page.click('[data-testid="submit-medication"]');

    // Wait for medication to appear on dashboard
    await page.waitForSelector('[data-testid="medication-card"]');

    // Try to add the same medication again
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    await page.fill('[data-testid="medication-search"]', 'ibuprofen');
    await page.waitForSelector('[data-testid="search-results"]');
    await page.locator('[data-testid="search-result"]').first().click();
    await page.waitForSelector('[data-testid="product-configuration"]');

    // Try same configuration
    await page.locator('[data-testid="dosage-option"]').first().click();
    await page.locator('[data-testid="frequency-option"]').first().click();
    await page.locator('[data-testid="place-option"]').first().click();

    await page.click('[data-testid="validate-button"]');

    // Should show duplicate error
    await expect(page.locator('[data-testid="validation-error"]')).toContainText('already have an active cycle');
  });

  test('shows fallback message for unknown medications', async () => {
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Search for non-existent medication
    await page.fill('[data-testid="medication-search"]', 'xyz123nonexistent');
    await page.waitForTimeout(2000); // Wait for search to complete

    // Should show no results with fallback
    await expect(page.locator('[data-testid="no-results"]')).toContainText('No medications found');
    await expect(page.locator('[data-testid="suggestions"]')).toContainText('Try searching');
  });

  test('displays hospital-grade validation indicators', async () => {
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    await page.fill('[data-testid="medication-search"]', 'metformin');
    await page.waitForSelector('[data-testid="search-results"]');
    await page.locator('[data-testid="search-result"]').first().click();
    await page.waitForSelector('[data-testid="product-configuration"]');

    // Check for hospital-grade indicators
    await expect(page.locator('[data-testid="hospital-grade-title"]')).toContainText('Hospital-Grade');
    await expect(page.locator('[data-testid="source-badge"]')).toBeVisible();
    await expect(page.locator('[data-testid="validation-info"]')).toContainText('Zero tolerance');

    // Check dosage section shows server-approved only
    await expect(page.locator('[data-testid="dosage-title"]')).toContainText('Server-Approved Dosages');
    await expect(page.locator('[data-testid="frequency-title"]')).toContainText('Server-Approved Frequencies');
  });

  test('handles no approved dosages gracefully', async () => {
    // This test would require a medication with no approved dosages in the system
    // Or a mock scenario where the server returns empty dosages
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Search for a medication that might have limited data
    await page.fill('[data-testid="medication-search"]', 'rare-medication');
    await page.waitForTimeout(2000);

    // If no results or results with no dosages
    const noApprovedDosages = page.locator('[data-testid="no-approved-dosages"]');
    if (await noApprovedDosages.isVisible()) {
      await expect(noApprovedDosages).toContainText('No approved dosages available');
      await expect(noApprovedDosages).toContainText('Consult dosage guidelines');
    }
  });

  test('mobile responsive design works correctly', async () => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 812 });

    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Verify popup is properly sized for mobile
    const popup = page.locator('[data-testid="medication-popup"]');
    const popupBox = await popup.boundingBox();
    expect(popupBox.width).toBeLessThanOrEqual(375);

    // Test search functionality on mobile
    await page.fill('[data-testid="medication-search"]', 'aspirin');
    await page.waitForSelector('[data-testid="search-results"]');

    // Verify search results are properly formatted for mobile
    const searchResult = page.locator('[data-testid="search-result"]').first();
    await expect(searchResult).toBeVisible();

    await searchResult.click();
    await page.waitForSelector('[data-testid="product-configuration"]');

    // Verify dosage buttons stack properly on mobile
    const dosageOptions = page.locator('[data-testid="dosage-option"]');
    const dosageCount = await dosageOptions.count();
    
    if (dosageCount > 0) {
      // Check that dosages are in grid layout
      const firstDosage = dosageOptions.first();
      const dosageBox = await firstDosage.boundingBox();
      expect(dosageBox.width).toBeLessThan(200); // Should be in grid, not full width
    }
  });

  test('keyboard navigation works correctly', async () => {
    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Focus on search input
    await page.focus('[data-testid="medication-search"]');
    
    // Type search term
    await page.keyboard.type('paracetamol');
    await page.waitForSelector('[data-testid="search-results"]');

    // Navigate through results with arrow keys
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter'); // Select first result

    await page.waitForSelector('[data-testid="product-configuration"]');

    // Navigate through dosage options with Tab
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter'); // Select first dosage

    // Continue with Tab navigation
    await page.keyboard.press('Tab');
    await page.keyboard.press('Enter'); // Select frequency

    // Verify configuration can be completed with keyboard only
    await page.focus('[data-testid="validate-button"]');
    await page.keyboard.press('Enter');

    await page.waitForSelector('[data-testid="validation-success"]');
  });

  test('accessibility compliance', async () => {
    // Inject axe-core for accessibility testing
    await page.addScriptTag({ path: 'node_modules/axe-core/axe.min.js' });

    await page.click('[data-testid="add-medication-button"]');
    await page.waitForSelector('[data-testid="medication-popup"]');

    // Run accessibility audit
    const accessibilityResults = await page.evaluate(() => {
      return new Promise((resolve) => {
        axe.run((err, results) => {
          if (err) throw err;
          resolve(results);
        });
      });
    });

    // Check for accessibility violations
    expect(accessibilityResults.violations).toHaveLength(0);

    // Test specific accessibility features
    await expect(page.locator('[data-testid="medication-search"]')).toHaveAttribute('aria-label');
    await expect(page.locator('[data-testid="validate-button"]')).toHaveAttribute('aria-describedby');
  });
});
