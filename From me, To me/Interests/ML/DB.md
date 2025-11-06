

1/ Reduce reads - Cache, cache, cache. Use read-thru and write-thru caching. Use Redis.

2/ Optimize slow reads - Add indexes. Fix your N+1 queries. Add limits. Sort responsibly. Alert on slow queries.

3/ Scale hardware - Add read replicas. Upgrade memory & I/O.

4/ Split up data - Shard when you have to. Partition writes when needed (you won’t need it).

Understand “good enough”. A single write instance is fine for most people.

And if it's not slow, leave it alone!
