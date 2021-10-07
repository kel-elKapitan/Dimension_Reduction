# dataset taken from https://www.kaggle.com/ananaymital/us-used-cars-dataset


import pandas as pd


# choose one type of vehicle as a sample becausae the dataset is over 9GB



iter_csv = pd.read_csv('used_cars_data.csv', iterator=True, chunksize=500)
raw = pd.concat([chunk[chunk['make_name'] == 'Jeep'] for chunk in iter_csv])


print(len(raw))
# Driver code for the application
