import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_csv(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)
    return data


# Data Summary Function
def Data_Summary(data):
    print("Data Summary for file: ", file)
    print("=====================================")
    print("First 5 rows")
    print(data.describe())
    print("=====================================")
    print("Data types")
    print(data.info())
    print("=====================================")
    print("Data count")
    print(data.count())
    print("=====================================")
    print("Missing values")
    print(data.isnull().sum())
    print("=====================================")
    print("Data shape")
    print(data.shape)
    print("=====================================")
    print("Unique values in each column")
    print(data.nunique())
    print("=====================================")

    # describe the last column
    print("Last column description")
    # print(data.iloc[:, -1].describe())
    # # Visualize the last column
    # temp_data = data.drop(data.index[0])
    # plt.figure(figsize=(7, 6))
    # plt.bar(temp_data.iloc[:, -1].unique(), temp_data.iloc[:, -1].value_counts())
    # plt.show(block=True)

    print("---------------------------------------------------------------------------------")

# Linear Interpolation but only for the columns with numerical values and same type of data
def LinearInterpolation(data):
    columns_names = ["bg-", "insulin-", "carbs-", "hr-", "steps-", "cals-"]
    last_column_name = columns_names[0]

    bg_df = data.filter(like='bg-', axis=1)
    bg_df = bg_df.interpolate(method='linear', axis=0)

    insulin_df = data.filter(like='insulin-', axis=1)
    insulin_df = insulin_df.interpolate(method='linear', axis=0)

    carbs_df = data.filter(like='carbs-', axis=1)
    carbs_df = carbs_df.interpolate(method='linear', axis=0)

    hr_df = data.filter(like='hr-', axis=1)
    hr_df = hr_df.interpolate(method='linear', axis=0)

    steps_df = data.filter(like='steps-', axis=1)
    steps_df = steps_df.interpolate(method='linear', axis=0)

    cals_df = data.filter(like='cals-', axis=1)
    cals_df = cals_df.interpolate(method='linear', axis=0)

    # the rest of the data that is not in the column names
    the_rest_df = data.drop(columns=bg_df.columns)
    the_rest_df = the_rest_df.drop(columns=insulin_df.columns)
    the_rest_df = the_rest_df.drop(columns=carbs_df.columns)
    the_rest_df = the_rest_df.drop(columns=hr_df.columns)
    the_rest_df = the_rest_df.drop(columns=steps_df.columns)
    the_rest_df = the_rest_df.drop(columns=cals_df.columns)

    data = pd.concat([bg_df, insulin_df, carbs_df, hr_df, steps_df, cals_df, the_rest_df], axis=1)

    return data


if __name__ == '__main__':
    file = "Data/train.csv"
    data = read_csv(file)

    data = LinearInterpolation(data)

    # save the data
    data.to_csv("Data/train_interpolated.csv", index=False)

    # Data_Summary(data)