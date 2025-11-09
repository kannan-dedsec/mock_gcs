from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field
from feast.types import Int64, Float32
from feast import FileSource

# Data source
stock_source = FileSource(
    path="../processed/stocks.parquet",
    timestamp_field="timestamp",
)

# Entity
stock = Entity(name="stock_id", join_keys=["stock_id"])

# Feature view
stock_fv = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="open", dtype=Float32),
        Field(name="high", dtype=Float32),
        Field(name="low", dtype=Float32),
        Field(name="close", dtype=Float32),
        Field(name="volume", dtype=Float32),
        Field(name="ma_10", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=stock_source,
)
