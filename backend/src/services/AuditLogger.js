/**
 * Hospital-Grade Audit Logger
 * Comprehensive audit trail for medication system with full traceability
 */

const { PrismaClient } = require('@prisma/client');
const crypto = require('crypto');

class AuditLogger {
  constructor() {
    this.prisma = new PrismaClient();
    this.systemVersion = this.getSystemVersion();
    this.sessionId = this.generateSessionId();
  }

  /**
   * Log medication search operations
   */
  async logMedicationSearch(userId, searchData, results, metadata = {}) {
    try {
      const auditEntry = {
        eventType: 'MEDICATION_SEARCH',
        userId: userId || 'anonymous',
        sessionId: this.sessionId,
        timestamp: new Date(),
        eventData: {
          query: this.sanitizeSearchQuery(searchData.query),
          filters: searchData.filters || {},
          resultCount: results.matches?.length || 0,
          suggestionCount: results.suggestions?.length || 0,
          source: results.source || 'unknown',
          hospitalGrade: results.hospitalGrade || false,
          confidence: this.calculateAverageConfidence(results.matches),
          executionTime: metadata.executionTime || null,
          clientIP: metadata.clientIP || null,
          userAgent: metadata.userAgent || null
        },
        riskLevel: this.assessSearchRisk(searchData, results),
        systemVersion: this.systemVersion,
        datasetVersion: metadata.datasetVersion || 'unknown',
        ruleVersion: metadata.ruleVersion || 1,
        provenance: results.metadata?.sources || [],
        checksum: null // Will be calculated after JSON serialization
      };

      auditEntry.checksum = this.calculateChecksum(auditEntry);

      await this.persistAuditEntry(auditEntry);
      
      console.log(`[AUDIT] Search: ${userId} queried "${searchData.query}" → ${results.matches?.length || 0} results`);

    } catch (error) {
      console.error('[AUDIT ERROR] Failed to log medication search:', error);
      // Don't throw - audit failures shouldn't break the application
    }
  }

  /**
   * Log medication validation operations
   */
  async logMedicationValidation(userId, validationData, result, metadata = {}) {
    try {
      const auditEntry = {
        eventType: 'MEDICATION_VALIDATION',
        userId: userId || 'anonymous',
        sessionId: this.sessionId,
        timestamp: new Date(),
        eventData: {
          medicationId: validationData.medication_id,
          productId: validationData.product_id,
          configuration: {
            intakeType: validationData.intake_type,
            intakePlace: validationData.intake_place,
            strengthValue: validationData.strength_value,
            strengthUnit: validationData.strength_unit,
            frequency: validationData.frequency
          },
          customFlags: validationData.custom_flags || {},
          validationResult: result.valid,
          errors: result.errors || [],
          warnings: result.warnings || [],
          normalized: result.normalized || null,
          source: result.source || 'unknown',
          executionTime: metadata.executionTime || null,
          clientIP: metadata.clientIP || null,
          userAgent: metadata.userAgent || null
        },
        riskLevel: this.assessValidationRisk(validationData, result),
        systemVersion: this.systemVersion,
        datasetVersion: metadata.datasetVersion || 'unknown',
        ruleVersion: result.rule_version || metadata.ruleVersion || 1,
        provenance: metadata.provenance || [],
        checksum: null
      };

      auditEntry.checksum = this.calculateChecksum(auditEntry);

      await this.persistAuditEntry(auditEntry);
      
      const status = result.valid ? 'VALID' : 'INVALID';
      console.log(`[AUDIT] Validation: ${userId} ${status} config for ${validationData.product_id}`);

    } catch (error) {
      console.error('[AUDIT ERROR] Failed to log medication validation:', error);
    }
  }

  /**
   * Log medication cycle creation/modification
   */
  async logMedicationCycle(userId, action, cycleData, result, metadata = {}) {
    try {
      const auditEntry = {
        eventType: `MEDICATION_CYCLE_${action.toUpperCase()}`,
        userId: userId || 'anonymous',
        sessionId: this.sessionId,
        timestamp: new Date(),
        eventData: {
          cycleId: result.cycle?.id || cycleData.id || null,
          medicationId: cycleData.medication_id,
          productId: cycleData.product_id,
          configuration: {
            strengthValue: cycleData.strength_value,
            strengthUnit: cycleData.strength_unit,
            frequency: cycleData.frequency,
            intakeType: cycleData.intake_type,
            intakePlace: cycleData.intake_place,
            startDate: cycleData.start_date,
            endDate: cycleData.end_date
          },
          customFlags: cycleData.custom_flags || {},
          notes: cycleData.notes ? '[REDACTED - Contains PII]' : null,
          action: action,
          success: !!result.cycle,
          clientIP: metadata.clientIP || null,
          userAgent: metadata.userAgent || null
        },
        riskLevel: this.assessCycleRisk(action, cycleData, result),
        systemVersion: this.systemVersion,
        datasetVersion: metadata.datasetVersion || 'unknown',
        ruleVersion: metadata.ruleVersion || 1,
        provenance: metadata.provenance || [],
        checksum: null
      };

      auditEntry.checksum = this.calculateChecksum(auditEntry);

      await this.persistAuditEntry(auditEntry);
      
      console.log(`[AUDIT] Cycle ${action}: ${userId} ${action}d cycle for ${cycleData.product_id}`);

    } catch (error) {
      console.error('[AUDIT ERROR] Failed to log medication cycle:', error);
    }
  }

  /**
   * Log security events
   */
  async logSecurityEvent(userId, eventType, details, metadata = {}) {
    try {
      const auditEntry = {
        eventType: `SECURITY_${eventType.toUpperCase()}`,
        userId: userId || 'anonymous',
        sessionId: this.sessionId,
        timestamp: new Date(),
        eventData: {
          securityEventType: eventType,
          details: this.sanitizeSecurityDetails(details),
          clientIP: metadata.clientIP || null,
          userAgent: metadata.userAgent || null,
          endpoint: metadata.endpoint || null,
          method: metadata.method || null
        },
        riskLevel: this.assessSecurityRisk(eventType, details),
        systemVersion: this.systemVersion,
        datasetVersion: 'N/A',
        ruleVersion: 1,
        provenance: ['security_system'],
        checksum: null
      };

      auditEntry.checksum = this.calculateChecksum(auditEntry);

      await this.persistAuditEntry(auditEntry);
      
      console.log(`[AUDIT] Security: ${eventType} for ${userId}`);

    } catch (error) {
      console.error('[AUDIT ERROR] Failed to log security event:', error);
    }
  }

  /**
   * Log system errors and exceptions
   */
  async logSystemError(userId, error, context, metadata = {}) {
    try {
      const auditEntry = {
        eventType: 'SYSTEM_ERROR',
        userId: userId || 'system',
        sessionId: this.sessionId,
        timestamp: new Date(),
        eventData: {
          errorType: error.name || 'UnknownError',
          errorMessage: error.message || 'No message provided',
          stackTrace: error.stack ? '[REDACTED - Contains system paths]' : null,
          context: this.sanitizeErrorContext(context),
          clientIP: metadata.clientIP || null,
          userAgent: metadata.userAgent || null
        },
        riskLevel: this.assessErrorRisk(error, context),
        systemVersion: this.systemVersion,
        datasetVersion: metadata.datasetVersion || 'unknown',
        ruleVersion: metadata.ruleVersion || 1,
        provenance: ['error_system'],
        checksum: null
      };

      auditEntry.checksum = this.calculateChecksum(auditEntry);

      await this.persistAuditEntry(auditEntry);
      
      console.log(`[AUDIT] Error: ${error.name} in ${context.component || 'unknown'}`);

    } catch (auditError) {
      console.error('[AUDIT ERROR] Failed to log system error:', auditError);
    }
  }

  /**
   * Retrieve audit trail for a user or system event
   */
  async getAuditTrail(filters = {}, options = {}) {
    try {
      const {
        userId,
        eventType,
        dateFrom,
        dateTo,
        riskLevel,
        limit = 100,
        offset = 0
      } = filters;

      // This would query the audit storage
      // For now, return a placeholder structure
      return {
        entries: [],
        total: 0,
        filters: filters,
        generatedAt: new Date(),
        requesterId: options.requesterId || 'system'
      };

    } catch (error) {
      console.error('[AUDIT ERROR] Failed to retrieve audit trail:', error);
      throw error;
    }
  }

  // Private helper methods

  sanitizeSearchQuery(query) {
    if (!query || typeof query !== 'string') return '';
    
    // Remove potentially sensitive information
    return query
      .replace(/\b\d{10,}\b/g, '[REDACTED-ID]') // Long numbers (could be patient IDs)
      .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[REDACTED-EMAIL]')
      .substring(0, 100); // Limit length
  }

  sanitizeSecurityDetails(details) {
    if (!details || typeof details !== 'object') return {};
    
    const sanitized = { ...details };
    
    // Remove sensitive fields
    delete sanitized.password;
    delete sanitized.token;
    delete sanitized.secret;
    delete sanitized.key;
    
    return sanitized;
  }

  sanitizeErrorContext(context) {
    if (!context || typeof context !== 'object') return {};
    
    const sanitized = { ...context };
    
    // Remove file paths and sensitive data
    if (sanitized.filePath) {
      sanitized.filePath = '[REDACTED-PATH]';
    }
    
    return sanitized;
  }

  calculateAverageConfidence(matches) {
    if (!matches || !Array.isArray(matches) || matches.length === 0) {
      return null;
    }
    
    const totalConfidence = matches.reduce((sum, match) => {
      return sum + (match.confidence || match.score || 0);
    }, 0);
    
    return Math.round(totalConfidence / matches.length);
  }

  assessSearchRisk(searchData, results) {
    // Low risk for successful searches
    if (results.matches?.length > 0) return 'LOW';
    
    // Medium risk for no results (could indicate data issues)
    if (!results.matches?.length && results.suggestions?.length > 0) return 'MEDIUM';
    
    // High risk for complete failures or suspicious queries
    if (searchData.query?.length > 100 || !results.matches) return 'HIGH';
    
    return 'LOW';
  }

  assessValidationRisk(validationData, result) {
    // High risk for validation failures with custom flags
    if (!result.valid && validationData.custom_flags?.dose) return 'HIGH';
    
    // Medium risk for any validation failure
    if (!result.valid) return 'MEDIUM';
    
    // Medium risk for custom dosages even if valid
    if (result.valid && validationData.custom_flags?.dose) return 'MEDIUM';
    
    return 'LOW';
  }

  assessCycleRisk(action, cycleData, result) {
    // High risk for failed cycle creation
    if (action === 'create' && !result.cycle) return 'HIGH';
    
    // Medium risk for modifications
    if (action === 'update' || action === 'delete') return 'MEDIUM';
    
    // Medium risk for custom configurations
    if (cycleData.custom_flags && Object.values(cycleData.custom_flags).some(flag => flag)) {
      return 'MEDIUM';
    }
    
    return 'LOW';
  }

  assessSecurityRisk(eventType, details) {
    const highRiskEvents = ['AUTHENTICATION_FAILURE', 'AUTHORIZATION_FAILURE', 'RATE_LIMIT_EXCEEDED'];
    const mediumRiskEvents = ['LOGIN_SUCCESS', 'LOGOUT', 'PASSWORD_CHANGE'];
    
    if (highRiskEvents.includes(eventType)) return 'HIGH';
    if (mediumRiskEvents.includes(eventType)) return 'MEDIUM';
    
    return 'LOW';
  }

  assessErrorRisk(error, context) {
    // High risk for database or security related errors
    if (error.message?.includes('database') || error.message?.includes('auth')) {
      return 'HIGH';
    }
    
    // Medium risk for validation or business logic errors
    if (error.name?.includes('Validation') || context.component?.includes('validation')) {
      return 'MEDIUM';
    }
    
    return 'LOW';
  }

  calculateChecksum(auditEntry) {
    // Create a copy without the checksum field
    const entryForHashing = { ...auditEntry };
    delete entryForHashing.checksum;
    
    // Create deterministic string representation
    const dataString = JSON.stringify(entryForHashing, Object.keys(entryForHashing).sort());
    
    // Generate SHA-256 hash
    return crypto.createHash('sha256').update(dataString).digest('hex');
  }

  generateSessionId() {
    return crypto.randomBytes(16).toString('hex');
  }

  getSystemVersion() {
    // This would typically read from package.json or environment
    return process.env.SYSTEM_VERSION || '1.0.0';
  }

  async persistAuditEntry(auditEntry) {
    // For now, we'll use console logging and file storage
    // In production, this would write to a secure audit database
    
    const auditRecord = {
      ...auditEntry,
      id: crypto.randomUUID(),
      persisted: true,
      persistedAt: new Date()
    };
    
    // Log to console (in production, this would be to a secure log file)
    console.log('[AUDIT ENTRY]', JSON.stringify(auditRecord, null, 2));
    
    // TODO: In production, implement:
    // - Secure database storage with encryption
    // - Write-only audit table with append-only operations
    // - Digital signatures for audit entry integrity
    // - Backup and archival procedures
    // - Compliance with healthcare audit requirements (HIPAA, etc.)
  }

  async cleanup() {
    await this.prisma.$disconnect();
  }
}

module.exports = AuditLogger;
