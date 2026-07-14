from feast import FileSource
from feast.data_format import ParquetFormat

insurance_source = FileSource(
    name="insurance_source",
    path="../../data/features_store/train_features.parquet",
    file_format=ParquetFormat(),
    event_timestamp_column="event_timestamp",
)