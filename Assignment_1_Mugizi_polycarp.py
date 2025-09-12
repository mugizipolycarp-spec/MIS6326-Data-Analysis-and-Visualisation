# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 19:57:40 2025

@author: polycarp.mugizi
"""
import pandas as pd

# importing a csv data file of World University Rankings from local disk D
file_path = r"D:\Msc\cwurData.csv"

# read the csv file
df = pd.read_csv(file_path)

# print out to confirm data in the file by reading first 10 records
print(df.head(10))

#function mean to get average of patents per university
def get_average_patents_per_institution(df):
    return df['patents'].mean()

print(get_average_patents_per_institution(df))

# function Count for number of universities ranked from USA
No_of_institutions_USA = df[df['country'] == 'USA']['country'].count()

print(f"Number of universities from USA: {No_of_institutions_USA}")

#number of top ranked universities
count = 0
for national_rank in df['national_rank']:
    if national_rank < 2:
        count += 1

print("Top Ranked institutions:", count)


# List of all countries
country_list = list(df['country'].unique())
print(country_list[:])

#tag national_rank level
def tag_level(national_rank):
    if national_rank < 2:
        return 'Top'
    elif national_rank < 5:
        return 'Medium'
    else:
        return 'Low'

df['level'] = df['national_rank'].apply(tag_level)
print(df[['national_rank', 'level']].head())


# Define an institution
class institution:
    def __init__(self, national_rank, quality_of_education):
        self.national_rank = national_rank
        self.squality_of_education = quality_of_education

    def show(self):
        print(f"{self.national_rank} has {self.squality_of_education}")

t = institution("institution", 9)
t.show()

