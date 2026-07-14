from feast import Field, FeatureView
from feast.types import Float32

from entities import person
from data_sources import insurance_source

schema = [
    Field(name=str(i), dtype=Float32)
    for i in range(77)
]

insurance_features = FeatureView(
    name="insurance_features",
    entities=[person],
    ttl=None,
    schema=schema,
    source=insurance_source,
    online=True,
)