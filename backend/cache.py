"""
High-Performance In-Memory Cache Engine with TTL & Eviction for RAIN-REPAIR X (PART 9).
Provides thread-safe caching of meteorological grid forecasts, district GeoJSON, and comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.cache")


@dataclass
class CacheEntry:
    value: Any
    expires_at_epoch: float
    created_at: datetime


class CacheManager:
    """
    Thread-safe in-memory cache supporting TTL expiration, prefix invalidation,
    and hit/miss telemetry.
    """

    def __init__(self, default_ttl_seconds: int = 180, max_size: int = 2000) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """
        Retrieves an item from cache.
        Returns (True, value) on hit, or (False, None) on miss/expired.
        """
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if now < entry.expires_at_epoch:
                    self.hits += 1
                    return True, entry.value
                else:
                    # Expired entry
                    del self._cache[key]
                    self.evictions += 1

            self.misses += 1
            return False, None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Sets a key-value pair in cache with specified TTL.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = datetime.now(timezone.utc).timestamp() + ttl
        with self._lock:
            # Simple capacity bound check
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                expires_at_epoch=expires_at,
                created_at=datetime.now(timezone.utc),
            )

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidates all cache keys matching the given prefix.
        Returns count of removed keys.
        """
        removed = 0
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]
                removed += 1
        logger.info("Invalidated %d cache keys matching prefix '%s'.", removed, prefix)
        return removed

    def clear(self) -> int:
        """Clears the entire cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        logger.info("Cleared entire cache (%d entries).", count)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry stats."""
        total_requests = self.hits + self.misses
        hit_ratio = (self.hits / total_requests) if total_requests > 0 else 0.0
        with self._lock:
            active_entries = len(self._cache)

        return {
            "active_entries": active_entries,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_ratio_pct": round(hit_ratio * 100.0, 2),
            "evictions": self.evictions,
        }


# Global cache singleton
cache_manager = CacheManager(default_ttl_seconds=300)
