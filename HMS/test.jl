using Parquet2, DataFrames, Tables

ds = Parquet2.Dataset("1000086677.parquet")
ds2 = Parquet2.Dataset("1000913311.parquet");

df = DataFrame(ds2);

df