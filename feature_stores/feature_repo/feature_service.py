from feast import FeatureService

from feature_views import insurance_features

insurance_service = FeatureService(
    name="insurance_service",
    features=[insurance_features],
)