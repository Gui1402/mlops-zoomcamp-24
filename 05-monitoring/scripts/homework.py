import pandas as pd
from evidently.metrics import ColumnQuantileMetric, ColumnCorrelationsMetric
from evidently.report import Report
from evidently import ColumnMapping



def preprocess(df):
    df["duration_min"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration_min = df.duration_min.apply(lambda td : float(td.total_seconds())/60)
    df = df[(df.duration_min >= 0) & (df.duration_min <= 60)]
    df = df[(df.passenger_count > 0) & (df.passenger_count <= 8)]
    return df


if __name__=="__main__":
    ## Reference data
    year = 2022
    month = 1
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet'
    reference_df = pd.read_parquet(filename)
    reference_df = preprocess(reference_df)
    ## prd_data
    year = 2024
    month = 3
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet'
    prod_df = pd.read_parquet(filename)
    print(f"Question 1: {len(prod_df)}")
    prod_df = preprocess(prod_df)

    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    report = Report(
        metrics=[
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        ColumnCorrelationsMetric(column_name="fare_amount")
    ])
    column_mapping = ColumnMapping(
        target=None,
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features
    )
    prod_df['day'] = prod_df['lpep_pickup_datetime'].dt.day

    maxq = 0
    for day in range(1, 32):
        current_df = prod_df[prod_df['day'] == day]
    
        report.run(
            reference_data=reference_df, 
            current_data=current_df, 
            column_mapping=column_mapping
        )
        result = report.as_dict()
        q05 = result["metrics"][0]["result"]["current"]["value"]
        #print(q05)
        
        if q05 > maxq:
            maxq = q05
        print(f"Quantile 0.5 Day {day}: {q05}")
        print(f"Max Quantile 0.5 {maxq}")