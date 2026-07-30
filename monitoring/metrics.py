from prometheus_client import Counter, Histogram


PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total number of predictions"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction processing latency"
)

API_REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests"
)


def record_prediction():
    """Increment prediction counter."""
    PREDICTION_COUNT.inc()


def record_api_request():
    """Increment API request counter."""
    API_REQUEST_COUNT.inc()


def observe_prediction_latency(seconds):
    """Record prediction latency."""
    PREDICTION_LATENCY.observe(seconds)