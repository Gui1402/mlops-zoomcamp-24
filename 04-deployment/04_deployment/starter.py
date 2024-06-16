
import pickle
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

base_folder = str(Path(__file__).parents[0])
with open(f'{base_folder}/model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def question1(pred):
    return pred.std()


def question2(df, output_file, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )



def main():
    parser = ArgumentParser(description="homework 4 script")
    parser.add_argument(
        '--year', 
        type=int, 
        required=True, 
        help="Year for the ride_id"
    )
    parser.add_argument(
        '--month', 
        type=int, 
        required=True, 
        help="Month for the ride_id"
    )
    args = parser.parse_args()
    year = args.year
    month = args.month
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    df["predictions"] = y_pred

    print(f"################## Question 1 ##################\nR:{question1(y_pred)}")
    print(f"################## Question 2 ##################\nR:{question2(df[["predictions"]], "question2.parquet", year, month)}")

if __name__ == "__main__":
    main()