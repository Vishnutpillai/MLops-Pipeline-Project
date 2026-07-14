from feast import Entity
from feast.value_type import ValueType

person = Entity(
    name="person",
    join_keys=["person_id"],
    value_type=ValueType.INT64,
)