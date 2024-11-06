
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
    # columns_names = ["bg-", "insulin-", "carbs-", "hr-", "steps-", "cals-"]

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
    the_rest_df = the_rest_df.drop(columns="id")

    data = pd.concat([the_rest_df, bg_df, insulin_df, carbs_df, hr_df, steps_df, cals_df], axis=1)

    return data


def prepare_data(data, train = True):
    # Fill missing values with 0
    global patient_ids
    data = data.fillna(0)

    if not train:
        patient_ids = X['p_num']

    # Change object to category if the column data type is object or column name contains "activity"
    for col in data.columns:
        if (data[col].dtype == 'object' and "id" not in col and "time" not in col) or "activity" in col:
            print(f"Before: {col} - {data[col].dtype}")
            # Assign the converted column back to the DataFrame
            data[col] = data[col].astype('category')
            print(f"After: {col} - {data[col].dtype}")
            data[col] = data[col].cat.codes

    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S')

    # Separate the column named "bg+1:00" from the data and delete it
    if train:
        y_train = data['bg+1:00']
        X_train = data.drop(columns=['bg+1:00'])
        return X_train, y_train
    else:
        X_Test = data
        X_Test['time_since'] = (X_Test['time'] - X_Test['time'].min()).dt.total_seconds()
        X_Test = X_Test.drop(columns=['time'])
        return X_Test, patient_ids

