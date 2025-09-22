/**
 * Smart data caching utility for enrichment results
 * Provides intelligent persistence with cache invalidation
 */

import { SpreadsheetData, CellData } from '../types';

interface CacheEntry {
  data: SpreadsheetData;
  timestamp: number;
  version: string;
  enrichmentStats: {
    enrichedCells: number;
    totalCells: number;
    lastEnrichmentTime: number;
  };
}

interface CacheConfig {
  maxAge: number; // Max age in milliseconds
  version: string; // App version for cache invalidation
  storageKey: string;
}

class DataCache {
  private config: CacheConfig;
  
  constructor(config: Partial<CacheConfig> = {}) {
    this.config = {
      maxAge: 24 * 60 * 60 * 1000, // 24 hours default
      version: '1.0.0',
      storageKey: 'tavily-enrichment-cache',
      ...config
    };
  }

  /**
   * Generate a stable cache key based on data content
   */
  private generateDataHash(data: SpreadsheetData): string {
    const content = JSON.stringify({
      headers: data.headers,
      rowCount: data.rows.length,
      // Sample first few cells to create a fingerprint
      sample: data.rows.slice(0, 3).map(row => 
        row.slice(0, 3).map(cell => typeof cell === 'string' ? cell : cell.value)
      )
    });
    
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < content.length; i++) {
      const char = content.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

  /**
   * Calculate enrichment statistics
   */
  private calculateEnrichmentStats(data: SpreadsheetData): CacheEntry['enrichmentStats'] {
    let enrichedCells = 0;
    let totalCells = 0;
    let lastEnrichmentTime = 0;

    data.rows.forEach(row => {
      row.forEach(cell => {
        totalCells++;
        const cellData = typeof cell === 'string' ? { value: cell } : cell;
        
        if (cellData.value && cellData.value.trim() !== '') {
          enrichedCells++;
          
          // Check if this cell has enrichment metadata (sources, timestamp, etc.)
          if ('sources' in cellData || 'enrichedAt' in cellData) {
            const enrichedAt = (cellData as any).enrichedAt;
            if (enrichedAt && enrichedAt > lastEnrichmentTime) {
              lastEnrichmentTime = enrichedAt;
            }
          }
        }
      });
    });

    return {
      enrichedCells,
      totalCells,
      lastEnrichmentTime: lastEnrichmentTime || Date.now()
    };
  }

  /**
   * Save data to cache with intelligent metadata
   */
  saveToCache(data: SpreadsheetData): void {
    try {
      const cacheEntry: CacheEntry = {
        data: this.deepClone(data),
        timestamp: Date.now(),
        version: this.config.version,
        enrichmentStats: this.calculateEnrichmentStats(data)
      };

      const dataHash = this.generateDataHash(data);
      const fullKey = `${this.config.storageKey}-${dataHash}`;
      
      localStorage.setItem(fullKey, JSON.stringify(cacheEntry));
      
      // Also save a reference to the latest cache
      localStorage.setItem(`${this.config.storageKey}-latest`, fullKey);
      
      console.log(`💾 Cached data: ${cacheEntry.enrichmentStats.enrichedCells}/${cacheEntry.enrichmentStats.totalCells} enriched cells`);
    } catch (error) {
      console.warn('Failed to save data to cache:', error);
    }
  }

  /**
   * Get the latest cached entry (if valid) without merging
   */
  getLatest(): { key: string; data: SpreadsheetData; timestamp: number } | null {
    try {
      const latestKey = localStorage.getItem(`${this.config.storageKey}-latest`);
      if (!latestKey) return null;

      const entry = this.loadCacheEntry(latestKey);
      if (entry && this.isCacheValid(entry)) {
        return { key: latestKey, data: this.deepClone(entry.data), timestamp: entry.timestamp };
      }
      return null;
    } catch (error) {
      console.warn('Failed to get latest cache:', error);
      return null;
    }
  }

  /**
   * Delete a specific cache entry by key and clean up the latest pointer if it matches
   */
  deleteCacheKey(key: string): void {
    try {
      localStorage.removeItem(key);
      const latestKey = localStorage.getItem(`${this.config.storageKey}-latest`);
      if (latestKey === key) {
        localStorage.removeItem(`${this.config.storageKey}-latest`);
      }
    } catch (error) {
      console.warn('Failed to delete cache key:', error);
    }
  }

  /**
   * Load data from cache with validation
   */
  loadFromCache(currentData?: SpreadsheetData): SpreadsheetData | null {
    try {
      // Try to load latest cache first
      const latestKey = localStorage.getItem(`${this.config.storageKey}-latest`);
      
      if (latestKey) {
        const cacheEntry = this.loadCacheEntry(latestKey);
        if (cacheEntry && this.isCacheValid(cacheEntry)) {
          // If we have current data, merge intelligently
          if (currentData && this.shouldMergeCache(cacheEntry, currentData)) {
            return this.mergeWithCache(currentData, cacheEntry.data);
          }
          
          console.log(`🔄 Restored from cache: ${cacheEntry.enrichmentStats.enrichedCells}/${cacheEntry.enrichmentStats.totalCells} enriched cells`);
          return cacheEntry.data;
        }
      }

      // Fallback: try to find compatible cache by structure
      if (currentData) {
        const dataHash = this.generateDataHash(currentData);
        const compatibleKey = `${this.config.storageKey}-${dataHash}`;
        const cacheEntry = this.loadCacheEntry(compatibleKey);
        
        if (cacheEntry && this.isCacheValid(cacheEntry)) {
          console.log(`🔄 Found compatible cache: ${cacheEntry.enrichmentStats.enrichedCells}/${cacheEntry.enrichmentStats.totalCells} enriched cells`);
          return this.mergeWithCache(currentData, cacheEntry.data);
        }
      }
      
      return null;
    } catch (error) {
      console.warn('Failed to load data from cache:', error);
      return null;
    }
  }

  /**
   * Load cache entry from storage
   */
  private loadCacheEntry(key: string): CacheEntry | null {
    try {
      const cached = localStorage.getItem(key);
      return cached ? JSON.parse(cached) : null;
    } catch {
      return null;
    }
  }

  /**
   * Check if cache entry is valid
   */
  private isCacheValid(cacheEntry: CacheEntry): boolean {
    const now = Date.now();
    const age = now - cacheEntry.timestamp;
    
    return (
      age < this.config.maxAge &&
      cacheEntry.version === this.config.version &&
      cacheEntry.data &&
      Array.isArray(cacheEntry.data.headers) &&
      Array.isArray(cacheEntry.data.rows)
    );
  }

  /**
   * Determine if cache should be merged with current data
   */
  private shouldMergeCache(cacheEntry: CacheEntry, currentData: SpreadsheetData): boolean {
    // Check if structures are compatible
    if (!this.areStructuresCompatible(cacheEntry.data, currentData)) {
      return false;
    }

    // Check if cache has meaningful enrichments
    return cacheEntry.enrichmentStats.enrichedCells > currentData.rows.length;
  }

  /**
   * Check if data structures are compatible for merging
   */
  private areStructuresCompatible(cached: SpreadsheetData, current: SpreadsheetData): boolean {
    return (
      cached.headers.length === current.headers.length &&
      cached.rows.length === current.rows.length &&
      cached.headers.every((header, i) => header === current.headers[i])
    );
  }

  /**
   * Intelligently merge current data with cached data
   */
  private mergeWithCache(current: SpreadsheetData, cached: SpreadsheetData): SpreadsheetData {
    const merged: SpreadsheetData = {
      headers: [...current.headers],
      rows: current.rows.map((currentRow, rowIndex) => {
        const cachedRow = cached.rows[rowIndex];
        if (!cachedRow) return [...currentRow];

        return currentRow.map((currentCell, colIndex) => {
          const cachedCell = cachedRow[colIndex];
          
          // Use cached cell if it has more data
          if (this.isCellMoreComplete(cachedCell, currentCell)) {
            return this.deepClone(cachedCell);
          }
          
          return this.deepClone(currentCell);
        });
      })
    };

    console.log('🔄 Merged current data with cached enrichments');
    return merged;
  }

  /**
   * Determine if cached cell is more complete than current cell
   */
  private isCellMoreComplete(cached: CellData | string, current: CellData | string): boolean {
    const cachedData = typeof cached === 'string' ? { value: cached } : cached;
    const currentData = typeof current === 'string' ? { value: current } : current;
    
    // Prefer cached if it has value and current doesn't
    if (cachedData.value?.trim() && !currentData.value?.trim()) {
      return true;
    }
    
    // Prefer cached if it has enrichment metadata
    if ('sources' in cachedData || 'enrichedAt' in cachedData) {
      return true;
    }
    
    return false;
  }

  /**
   * Clear all cache entries
   */
  clearCache(): void {
    try {
      const keys = Object.keys(localStorage).filter(key => 
        key.startsWith(this.config.storageKey)
      );
      
      keys.forEach(key => localStorage.removeItem(key));
      console.log(`🗑️ Cleared ${keys.length} cache entries`);
    } catch (error) {
      console.warn('Failed to clear cache:', error);
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { entries: number; totalSize: number; latestEnrichment?: Date } {
    let entries = 0;
    let totalSize = 0;
    let latestEnrichment: Date | undefined;

    try {
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith(this.config.storageKey)) {
          entries++;
          const value = localStorage.getItem(key);
          if (value) {
            totalSize += value.length;
            
            try {
              const entry: CacheEntry = JSON.parse(value);
              if (entry.enrichmentStats?.lastEnrichmentTime) {
                const enrichmentDate = new Date(entry.enrichmentStats.lastEnrichmentTime);
                if (!latestEnrichment || enrichmentDate > latestEnrichment) {
                  latestEnrichment = enrichmentDate;
                }
              }
            } catch {}
          }
        }
      });
    } catch (error) {
      console.warn('Failed to calculate cache stats:', error);
    }

    return { entries, totalSize, latestEnrichment };
  }

  /**
   * Deep clone utility
   */
  private deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime()) as any;
    if (obj instanceof Array) return obj.map(item => this.deepClone(item)) as any;
    if (typeof obj === 'object') {
      const cloned: any = {};
      Object.keys(obj).forEach(key => {
        cloned[key] = this.deepClone((obj as any)[key]);
      });
      return cloned;
    }
    return obj;
  }
}

// Export singleton instance
export const dataCache = new DataCache({
  version: '1.0.0',
  maxAge: 24 * 60 * 60 * 1000, // 24 hours
  storageKey: 'tavily-enrichment-cache'
});

export default DataCache;