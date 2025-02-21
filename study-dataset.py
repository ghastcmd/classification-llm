import pandas as pd

df = pd.read_csv('./dataset/cleaned-dataset.csv')

df['group_type'] = df['group'] + '/' + df['type']

count_class = df['group_type'].value_counts()
print(count_class)
count_class = count_class.sort_index()
print(count_class)