/**
 * Hospital-Grade Safety Monitor
 * Real-time safety monitoring and alerting for medication system
 */

const AuditLogger = require('./AuditLogger');

class SafetyMonitor {
  constructor() {
    this.auditLogger = new AuditLogger();
    this.alertThresholds = this.getAlertThresholds();
    this.safetyMetrics = this.initializeSafetyMetrics();
    this.activeAlerts = new Map();
    this.monitoringInterval = null;
  }

  /**
   * Initialize safety monitoring
   */
  async initialize() {
    console.log('🛡️ Initializing Hospital-Grade Safety Monitor...');
    
    // Start continuous monitoring
    this.startContinuousMonitoring();
    
    // Initialize safety metrics
    this.resetSafetyMetrics();
    
    console.log('✅ Safety Monitor Active');
  }

  /**
   * Monitor medication validation for safety patterns
   */
  async monitorValidation(userId, validationData, result, metadata = {}) {
    try {
      const safetyEvents = [];
      
      // 1. Check for dangerous dose patterns
      const doseEvent = this.checkDangerousDoses(validationData, result);
      if (doseEvent) safetyEvents.push(doseEvent);
      
      // 2. Check for contraindication patterns
      const contraindicationEvent = this.checkContraindications(userId, validationData, result);
      if (contraindicationEvent) safetyEvents.push(contraindicationEvent);
      
      // 3. Check for suspicious validation patterns
      const suspiciousEvent = this.checkSuspiciousPatterns(userId, validationData, result);
      if (suspiciousEvent) safetyEvents.push(suspiciousEvent);
      
      // 4. Check for repeated failures
      const failureEvent = this.checkRepeatedFailures(userId, validationData, result);
      if (failureEvent) safetyEvents.push(failureEvent);
      
      // Process any safety events
      for (const event of safetyEvents) {
        await this.processSafetyEvent(event, userId, validationData, result, metadata);
      }
      
      // Update metrics
      this.updateValidationMetrics(result.valid, safetyEvents.length > 0);
      
    } catch (error) {
      console.error('[SAFETY ERROR] Failed to monitor validation:', error);
      await this.auditLogger.logSystemError(userId, error, { component: 'SafetyMonitor' });
    }
  }

  /**
   * Monitor medication cycles for safety issues
   */
  async monitorCycle(userId, action, cycleData, result, metadata = {}) {
    try {
      const safetyEvents = [];
      
      // 1. Check for dangerous drug combinations
      if (action === 'create') {
        const combinationEvent = await this.checkDrugCombinations(userId, cycleData);
        if (combinationEvent) safetyEvents.push(combinationEvent);
      }
      
      // 2. Check for excessive medication load
      const overloadEvent = await this.checkMedicationOverload(userId, cycleData);
      if (overloadEvent) safetyEvents.push(overloadEvent);
      
      // 3. Check for suspicious timing patterns
      const timingEvent = this.checkSuspiciousTiming(userId, cycleData);
      if (timingEvent) safetyEvents.push(timingEvent);
      
      // Process safety events
      for (const event of safetyEvents) {
        await this.processSafetyEvent(event, userId, cycleData, result, metadata);
      }
      
      // Update metrics
      this.updateCycleMetrics(action, safetyEvents.length > 0);
      
    } catch (error) {
      console.error('[SAFETY ERROR] Failed to monitor cycle:', error);
      await this.auditLogger.logSystemError(userId, error, { component: 'SafetyMonitor' });
    }
  }

  /**
   * Monitor system performance and availability
   */
  async monitorSystemHealth() {
    try {
      const healthEvents = [];
      
      // 1. Check response times
      const responseTimeEvent = this.checkResponseTimes();
      if (responseTimeEvent) healthEvents.push(responseTimeEvent);
      
      // 2. Check error rates
      const errorRateEvent = this.checkErrorRates();
      if (errorRateEvent) healthEvents.push(errorRateEvent);
      
      // 3. Check validation accuracy
      const accuracyEvent = this.checkValidationAccuracy();
      if (accuracyEvent) healthEvents.push(accuracyEvent);
      
      // 4. Check data source availability
      const dataSourceEvent = this.checkDataSources();
      if (dataSourceEvent) healthEvents.push(dataSourceEvent);
      
      // Process health events
      for (const event of healthEvents) {
        await this.processHealthEvent(event);
      }
      
      // Update health metrics
      this.updateHealthMetrics(healthEvents);
      
    } catch (error) {
      console.error('[SAFETY ERROR] Failed to monitor system health:', error);
      await this.auditLogger.logSystemError('system', error, { component: 'SafetyMonitor' });
    }
  }

  // Safety Check Methods

  checkDangerousDoses(validationData, result) {
    // Check for extremely high custom doses
    if (validationData.custom_flags?.dose && validationData.strength_value) {
      const dose = parseFloat(validationData.strength_value);
      const unit = validationData.strength_unit;
      
      // Define dangerous dose thresholds (these would be configured per medication)
      const dangerousThresholds = {
        'mg': 10000,  // > 10g for most medications
        'mcg': 10000, // > 10mg for micrograms
        'IU': 100000  // > 100k IU
      };
      
      if (dose > (dangerousThresholds[unit] || dangerousThresholds['mg'])) {
        return {
          type: 'DANGEROUS_DOSE',
          severity: 'CRITICAL',
          message: `Extremely high custom dose: ${dose} ${unit}`,
          data: { dose, unit, threshold: dangerousThresholds[unit] }
        };
      }
    }
    
    return null;
  }

  checkContraindications(userId, validationData, result) {
    // This would check against known contraindications
    // For now, implement basic checks
    
    if (!result.valid && result.errors) {
      const contraindicationErrors = result.errors.filter(error => 
        error.message?.toLowerCase().includes('contraindic') ||
        error.message?.toLowerCase().includes('dangerous') ||
        error.message?.toLowerCase().includes('unsafe')
      );
      
      if (contraindicationErrors.length > 0) {
        return {
          type: 'CONTRAINDICATION_DETECTED',
          severity: 'HIGH',
          message: 'Contraindication detected in validation',
          data: { errors: contraindicationErrors }
        };
      }
    }
    
    return null;
  }

  checkSuspiciousPatterns(userId, validationData, result) {
    // Track validation attempts for patterns
    const userKey = `validation_attempts_${userId}`;
    
    if (!this.safetyMetrics.userPatterns.has(userKey)) {
      this.safetyMetrics.userPatterns.set(userKey, {
        attempts: 0,
        failures: 0,
        customAttempts: 0,
        lastAttempt: null
      });
    }
    
    const pattern = this.safetyMetrics.userPatterns.get(userKey);
    pattern.attempts++;
    pattern.lastAttempt = new Date();
    
    if (!result.valid) pattern.failures++;
    if (validationData.custom_flags && Object.values(validationData.custom_flags).some(flag => flag)) {
      pattern.customAttempts++;
    }
    
    // Check for suspicious patterns
    const timeWindow = 10 * 60 * 1000; // 10 minutes
    const recentAttempts = pattern.attempts; // Simplified - would track time-based windows
    
    if (recentAttempts > 20) {
      return {
        type: 'EXCESSIVE_VALIDATION_ATTEMPTS',
        severity: 'MEDIUM',
        message: `User ${userId} has made ${recentAttempts} validation attempts`,
        data: { pattern }
      };
    }
    
    if (pattern.customAttempts > 10) {
      return {
        type: 'EXCESSIVE_CUSTOM_CONFIGURATIONS',
        severity: 'HIGH',
        message: `User ${userId} has attempted ${pattern.customAttempts} custom configurations`,
        data: { pattern }
      };
    }
    
    return null;
  }

  checkRepeatedFailures(userId, validationData, result) {
    if (result.valid) return null;
    
    const userKey = `failures_${userId}`;
    
    if (!this.safetyMetrics.failurePatterns.has(userKey)) {
      this.safetyMetrics.failurePatterns.set(userKey, {
        consecutiveFailures: 0,
        lastFailureTime: null
      });
    }
    
    const failurePattern = this.safetyMetrics.failurePatterns.get(userKey);
    failurePattern.consecutiveFailures++;
    failurePattern.lastFailureTime = new Date();
    
    if (failurePattern.consecutiveFailures >= 5) {
      return {
        type: 'REPEATED_VALIDATION_FAILURES',
        severity: 'MEDIUM',
        message: `User ${userId} has ${failurePattern.consecutiveFailures} consecutive validation failures`,
        data: { failurePattern }
      };
    }
    
    return null;
  }

  async checkDrugCombinations(userId, cycleData) {
    // This would check against a drug interaction database
    // For now, implement basic placeholder logic
    
    // TODO: Implement comprehensive drug interaction checking
    // - Query active cycles for the user
    // - Check against drug interaction database
    // - Identify potentially dangerous combinations
    
    return null;
  }

  async checkMedicationOverload(userId, cycleData) {
    // Check if user has too many active medications
    
    // TODO: Query user's active medication cycles
    // const activeCycles = await this.getUserActiveCycles(userId);
    
    // Placeholder logic
    const activeCycleCount = 0; // Would be actual count
    
    if (activeCycleCount > 15) {
      return {
        type: 'MEDICATION_OVERLOAD',
        severity: 'HIGH',
        message: `User ${userId} has ${activeCycleCount} active medications`,
        data: { activeCycleCount }
      };
    }
    
    return null;
  }

  checkSuspiciousTiming(userId, cycleData) {
    // Check for suspicious timing patterns (e.g., multiple cycles created rapidly)
    
    const now = new Date();
    const userKey = `timing_${userId}`;
    
    if (!this.safetyMetrics.timingPatterns.has(userKey)) {
      this.safetyMetrics.timingPatterns.set(userKey, {
        recentCycles: [],
        lastCycleTime: null
      });
    }
    
    const pattern = this.safetyMetrics.timingPatterns.get(userKey);
    pattern.recentCycles.push(now);
    pattern.lastCycleTime = now;
    
    // Keep only cycles from last hour
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
    pattern.recentCycles = pattern.recentCycles.filter(time => time > oneHourAgo);
    
    if (pattern.recentCycles.length > 10) {
      return {
        type: 'RAPID_CYCLE_CREATION',
        severity: 'MEDIUM',
        message: `User ${userId} created ${pattern.recentCycles.length} cycles in the last hour`,
        data: { pattern }
      };
    }
    
    return null;
  }

  // System Health Checks

  checkResponseTimes() {
    const avgResponseTime = this.safetyMetrics.performance.averageResponseTime;
    
    if (avgResponseTime > this.alertThresholds.responseTime.critical) {
      return {
        type: 'CRITICAL_RESPONSE_TIME',
        severity: 'CRITICAL',
        message: `Average response time is ${avgResponseTime}ms`,
        data: { avgResponseTime, threshold: this.alertThresholds.responseTime.critical }
      };
    }
    
    if (avgResponseTime > this.alertThresholds.responseTime.warning) {
      return {
        type: 'HIGH_RESPONSE_TIME',
        severity: 'MEDIUM',
        message: `Average response time is ${avgResponseTime}ms`,
        data: { avgResponseTime, threshold: this.alertThresholds.responseTime.warning }
      };
    }
    
    return null;
  }

  checkErrorRates() {
    const errorRate = this.safetyMetrics.performance.errorRate;
    
    if (errorRate > this.alertThresholds.errorRate.critical) {
      return {
        type: 'CRITICAL_ERROR_RATE',
        severity: 'CRITICAL',
        message: `Error rate is ${errorRate}%`,
        data: { errorRate, threshold: this.alertThresholds.errorRate.critical }
      };
    }
    
    if (errorRate > this.alertThresholds.errorRate.warning) {
      return {
        type: 'HIGH_ERROR_RATE',
        severity: 'MEDIUM',
        message: `Error rate is ${errorRate}%`,
        data: { errorRate, threshold: this.alertThresholds.errorRate.warning }
      };
    }
    
    return null;
  }

  checkValidationAccuracy() {
    const accuracy = this.safetyMetrics.validation.accuracy;
    
    if (accuracy < this.alertThresholds.validationAccuracy.critical) {
      return {
        type: 'LOW_VALIDATION_ACCURACY',
        severity: 'CRITICAL',
        message: `Validation accuracy is ${accuracy}%`,
        data: { accuracy, threshold: this.alertThresholds.validationAccuracy.critical }
      };
    }
    
    return null;
  }

  checkDataSources() {
    // Check if critical data sources are available
    const unavailableSources = this.safetyMetrics.dataSources.filter(source => !source.available);
    
    if (unavailableSources.length > 0) {
      const severity = unavailableSources.some(s => s.critical) ? 'CRITICAL' : 'MEDIUM';
      
      return {
        type: 'DATA_SOURCE_UNAVAILABLE',
        severity,
        message: `${unavailableSources.length} data sources unavailable`,
        data: { unavailableSources }
      };
    }
    
    return null;
  }

  // Event Processing

  async processSafetyEvent(event, userId, data, result, metadata) {
    try {
      // Log the safety event
      await this.auditLogger.logSecurityEvent(userId, `SAFETY_${event.type}`, {
        severity: event.severity,
        message: event.message,
        eventData: event.data
      }, metadata);
      
      // Check if this should trigger an alert
      if (this.shouldTriggerAlert(event)) {
        await this.triggerAlert(event, userId, data, result, metadata);
      }
      
      // Update safety metrics
      this.updateSafetyEventMetrics(event);
      
      console.log(`[SAFETY] ${event.severity}: ${event.message}`);
      
    } catch (error) {
      console.error('[SAFETY ERROR] Failed to process safety event:', error);
    }
  }

  async processHealthEvent(event) {
    try {
      // Log the health event
      await this.auditLogger.logSystemError('system', new Error(event.message), {
        component: 'HealthMonitor',
        eventType: event.type,
        severity: event.severity
      });
      
      // Trigger alert if necessary
      if (this.shouldTriggerAlert(event)) {
        await this.triggerSystemAlert(event);
      }
      
      console.log(`[HEALTH] ${event.severity}: ${event.message}`);
      
    } catch (error) {
      console.error('[SAFETY ERROR] Failed to process health event:', error);
    }
  }

  shouldTriggerAlert(event) {
    // Don't trigger duplicate alerts
    const alertKey = `${event.type}_${event.severity}`;
    
    if (this.activeAlerts.has(alertKey)) {
      const lastAlert = this.activeAlerts.get(alertKey);
      const timeSinceLastAlert = Date.now() - lastAlert;
      
      // Don't send same alert more than once per hour
      if (timeSinceLastAlert < 60 * 60 * 1000) {
        return false;
      }
    }
    
    // Trigger alerts for medium severity and above
    return ['MEDIUM', 'HIGH', 'CRITICAL'].includes(event.severity);
  }

  async triggerAlert(event, userId, data, result, metadata) {
    const alert = {
      id: this.generateAlertId(),
      timestamp: new Date(),
      type: event.type,
      severity: event.severity,
      message: event.message,
      userId: userId,
      data: event.data,
      acknowledged: false
    };
    
    // Store alert
    const alertKey = `${event.type}_${event.severity}`;
    this.activeAlerts.set(alertKey, Date.now());
    
    // In production, this would:
    // - Send notifications to healthcare staff
    // - Integration with hospital alert systems
    // - Email/SMS notifications for critical issues
    // - Dashboard alerts for monitoring staff
    
    console.log(`🚨 [ALERT] ${alert.severity}: ${alert.message}`);
    
    // TODO: Implement actual alerting mechanisms
  }

  async triggerSystemAlert(event) {
    const alert = {
      id: this.generateAlertId(),
      timestamp: new Date(),
      type: event.type,
      severity: event.severity,
      message: event.message,
      system: true,
      acknowledged: false
    };
    
    console.log(`🚨 [SYSTEM ALERT] ${alert.severity}: ${alert.message}`);
    
    // TODO: Implement system alerting mechanisms
  }

  // Utility Methods

  getAlertThresholds() {
    return {
      responseTime: {
        warning: 2000,    // 2 seconds
        critical: 5000    // 5 seconds
      },
      errorRate: {
        warning: 5,       // 5%
        critical: 10      // 10%
      },
      validationAccuracy: {
        critical: 95      // Below 95%
      }
    };
  }

  initializeSafetyMetrics() {
    return {
      validation: {
        total: 0,
        valid: 0,
        invalid: 0,
        accuracy: 100,
        safetyEvents: 0
      },
      cycles: {
        created: 0,
        updated: 0,
        deleted: 0,
        safetyEvents: 0
      },
      performance: {
        averageResponseTime: 0,
        errorRate: 0,
        totalRequests: 0,
        errorCount: 0
      },
      dataSources: [
        { name: 'NHS', available: true, critical: true },
        { name: 'RxNorm', available: true, critical: false },
        { name: 'Local', available: true, critical: true }
      ],
      userPatterns: new Map(),
      failurePatterns: new Map(),
      timingPatterns: new Map()
    };
  }

  resetSafetyMetrics() {
    this.safetyMetrics = this.initializeSafetyMetrics();
  }

  updateValidationMetrics(isValid, hasSafetyEvent) {
    this.safetyMetrics.validation.total++;
    
    if (isValid) {
      this.safetyMetrics.validation.valid++;
    } else {
      this.safetyMetrics.validation.invalid++;
    }
    
    if (hasSafetyEvent) {
      this.safetyMetrics.validation.safetyEvents++;
    }
    
    this.safetyMetrics.validation.accuracy = Math.round(
      (this.safetyMetrics.validation.valid / this.safetyMetrics.validation.total) * 100
    );
  }

  updateCycleMetrics(action, hasSafetyEvent) {
    this.safetyMetrics.cycles[action === 'create' ? 'created' : action === 'update' ? 'updated' : 'deleted']++;
    
    if (hasSafetyEvent) {
      this.safetyMetrics.cycles.safetyEvents++;
    }
  }

  updateHealthMetrics(healthEvents) {
    // Update performance metrics based on health events
    const criticalEvents = healthEvents.filter(e => e.severity === 'CRITICAL');
    
    if (criticalEvents.length > 0) {
      this.safetyMetrics.performance.errorCount += criticalEvents.length;
    }
  }

  updateSafetyEventMetrics(event) {
    // Track safety event patterns for trend analysis
    // This would be used for reporting and continuous improvement
  }

  generateAlertId() {
    return `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  startContinuousMonitoring() {
    // Monitor system health every 5 minutes
    this.monitoringInterval = setInterval(() => {
      this.monitorSystemHealth().catch(error => {
        console.error('[SAFETY ERROR] Continuous monitoring failed:', error);
      });
    }, 5 * 60 * 1000);
  }

  stopContinuousMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  async getSafetyReport() {
    return {
      timestamp: new Date(),
      metrics: this.safetyMetrics,
      activeAlerts: Array.from(this.activeAlerts.entries()).map(([key, time]) => ({
        key,
        lastTriggered: new Date(time)
      })),
      thresholds: this.alertThresholds,
      status: this.calculateOverallSafetyStatus()
    };
  }

  calculateOverallSafetyStatus() {
    const { validation, performance } = this.safetyMetrics;
    
    if (validation.accuracy < 95 || performance.errorRate > 10) {
      return 'CRITICAL';
    }
    
    if (validation.accuracy < 98 || performance.errorRate > 5) {
      return 'WARNING';
    }
    
    return 'HEALTHY';
  }

  async cleanup() {
    this.stopContinuousMonitoring();
    await this.auditLogger.cleanup();
  }
}

module.exports = SafetyMonitor;
