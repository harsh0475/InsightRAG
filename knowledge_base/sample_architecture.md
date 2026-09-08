# System Architecture and Distributed Cache Guidelines

## 1. Executive Overview
This document specifies the distributed caching policy and architectural standards for high-throughput enterprise services. All service teams must conform to these latency bounds and eviction strategies to prevent cache stampedes and cascading database outages.

## 2. Cache Invalidation Patterns
Cache invalidation is notoriously challenging in distributed systems. We mandate two primary patterns:
1. Write-Through Caching: When an update occurs, the application writes to the primary relational database and synchronously updates the Redis cache before returning success to the client. This guarantees strong read-after-write consistency at the expense of write latency.
2. Cache-Aside (Lazy Loading): The application queries the cache first. If a cache miss occurs, the application reads from the database, populates the cache with a time-to-live (TTL), and returns the payload.

## 3. TTL and Stampede Mitigation
To avoid cache stampedes when hot keys expire simultaneously:
- Jittered Expiration: Add a pseudo-random jitter between 5% and 15% to the base TTL. For a 3600-second TTL, the actual expiration should range between 3780 and 4140 seconds.
- Probabilistic Early Recomputation: Services may use the XFetch algorithm to refresh cache entries in the background prior to strict expiration.

## 4. Disaster Recovery and Replication
All Redis clusters must be deployed across at least three Availability Zones with asynchronous replication. In the event of primary node failure, automated sentinel failover must achieve quorum within 15 seconds.

