"""Histogram metrics."""
def observe(hist, bucket, count=1):
    """Add observation to bucket."""
    hist[bucket] = hist.get(bucket, 0) + count
    return hist

def get_bucket_count(hist, bucket):
    """Get count for bucket."""
    return hist.get(bucket, 0)

def histogram_buckets(hist):
    """Return sorted bucket names."""
    return sorted(hist.keys())