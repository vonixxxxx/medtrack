const NodeCache = require('node-cache');

/**
 * Drug data caching utility
 * Provides short-term in-memory caching for API responses
 * No persistent storage of bulk datasets
 */

class DrugCache {
  constructor() {
    // In-memory cache with TTL (Time To Live)
    this.cache = new NodeCache({
      stdTTL: 3600, // 1 hour default TTL
      checkperiod: 600, // Check for expired keys every 10 minutes
      maxKeys: 1000, // Maximum number of keys in cache
      useClones: false // Don't clone objects for better performance
    });

    // Cache statistics
    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0,
      deletes: 0
    };
  }

  /**
   * Generate cache key for search queries
   */
  generateSearchKey(query, limit = 10) {
    return `search:${query.toLowerCase().trim()}:${limit}`;
  }

  /**
   * Generate cache key for drug details
   */
  generateDetailsKey(source, id) {
    return `details:${source}:${id}`;
  }

  /**
   * Get cached search results
   */
  getSearchResults(query, limit = 10) {
    const key = this.generateSearchKey(query, limit);
    const result = this.cache.get(key);
    
    if (result !== undefined) {
      this.stats.hits++;
      return result;
    }
    
    this.stats.misses++;
    return null;
  }

  /**
   * Cache search results
   */
  setSearchResults(query, limit, results) {
    const key = this.generateSearchKey(query, limit);
    
    // Set cache with shorter TTL for search results (30 minutes)
    this.cache.set(key, results, 1800);
    this.stats.sets++;
    
    return true;
  }

  /**
   * Get cached drug details
   */
  getDrugDetails(source, id) {
    const key = this.generateDetailsKey(source, id);
    const result = this.cache.get(key);
    
    if (result !== undefined) {
      this.stats.hits++;
      return result;
    }
    
    this.stats.misses++;
    return null;
  }

  /**
   * Cache drug details
   */
  setDrugDetails(source, id, details) {
    const key = this.generateDetailsKey(source, id);
    
    // Set cache with longer TTL for drug details (2 hours)
    this.cache.set(key, details, 7200);
    this.stats.sets++;
    
    return true;
  }

  /**
   * Invalidate cache for a specific search query
   */
  invalidateSearch(query, limit = 10) {
    const key = this.generateSearchKey(query, limit);
    this.cache.del(key);
    this.stats.deletes++;
  }

  /**
   * Invalidate cache for a specific drug
   */
  invalidateDrug(source, id) {
    const key = this.generateDetailsKey(source, id);
    this.cache.del(key);
    this.stats.deletes++;
  }

  /**
   * Clear all cache entries
   */
  clearAll() {
    this.cache.flushAll();
    this.stats.deletes += this.cache.getStats().keys;
  }

  /**
   * Get cache statistics
   */
  getStats() {
    const cacheStats = this.cache.getStats();
    return {
      ...this.stats,
      cacheKeys: cacheStats.keys,
      cacheHits: cacheStats.hits,
      cacheMisses: cacheStats.misses,
      cacheSize: cacheStats.ksize,
      cacheVsize: cacheStats.vsize
    };
  }

  /**
   * Get cache health status
   */
  getHealth() {
    const stats = this.getStats();
    const hitRate = stats.cacheHits + stats.cacheMisses > 0 
      ? (stats.cacheHits / (stats.cacheHits + stats.cacheMisses)) * 100 
      : 0;
    
    return {
      status: 'healthy',
      hitRate: hitRate.toFixed(2) + '%',
      totalRequests: stats.cacheHits + stats.cacheMisses,
      cacheSize: stats.cacheKeys,
      memoryUsage: process.memoryUsage()
    };
  }

  /**
   * Preload cache with frequently searched drugs
   * This is NOT bulk data storage - just common searches
   */
  async preloadCommonSearches() {
    const commonSearches = [
      'aspirin',
      'ibuprofen',
      'paracetamol',
      'metformin',
      'omeprazole',
      'amlodipine',
      'atorvastatin',
      'metoprolol',
      'lisinopril',
      'losartan'
    ];

    // Note: This would call the actual APIs, not store bulk data
    // Implementation depends on your specific needs
    console.log('Common searches available for preloading:', commonSearches);
  }

  /**
   * Clean up expired cache entries
   */
  cleanup() {
    // node-cache automatically handles cleanup, no need to call prune
    // Just log the cleanup for debugging
    console.log('Cache cleanup completed');
  }

  /**
   * Set custom TTL for specific keys
   */
  setWithCustomTTL(key, value, ttl) {
    this.cache.set(key, value, ttl);
    this.stats.sets++;
  }

  /**
   * Check if a key exists in cache
   */
  has(key) {
    return this.cache.has(key);
  }

  /**
   * Get all keys in cache (for debugging)
   */
  getKeys() {
    return this.cache.keys();
  }

  /**
   * Get cache size information
   */
  getSize() {
    const stats = this.cache.getStats();
    return {
      keys: stats.keys,
      ksize: stats.ksize,
      vsize: stats.vsize
    };
  }
}

// Create singleton instance
const drugCache = new DrugCache();

// Cleanup expired entries every 5 minutes
setInterval(() => {
  drugCache.cleanup();
}, 5 * 60 * 1000);

module.exports = drugCache;
