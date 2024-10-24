import pandas as pd

def calculate_distance_matrix(df)-> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    df = pd.read_csv(file_path)
    locations = pd.concat([df['id_start'], df['id_end']]).unique()
    distance_matrix = pd.DataFrame(float('nan'), index=locations, columns=locations)
    for index, row in df.iterrows(): 
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']

    distance_matrix.fillna(float('inf'), inplace=True)
    for location in locations:
        distance_matrix.at[location, location] = 0
    for k in locations:
        for i in locations:
            for j in locations:
                distance_matrix.at[i, j] = min(distance_matrix.at[i, j], 
                                               distance_matrix.at[i, k] + distance_matrix.at[k, j])

    return distance_matrix

file_path ="datasets\dataset-2.csv"
distance_matrix = calculate_distance_matrix(file_path)
print(distance_matrix)

def unroll_distance_matrix(df)->pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance_matrix.at[id_start, id_end]
                })
    unrolled_df = pd.DataFrame(unrolled_data)
    return unrolled_df
distance_matrix = calculate_distance_matrix(file_path)
unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    distances = unrolled_df[unrolled_df['id_start'] == reference_id]['distance']
    
    if distances.empty:  
        return [] 
    avg_distance = distances.mean()
    lower_bound = avg_distance * 0.9
    upper_bound = avg_distance * 1.1
    filtered_ids = unrolled_df[
        (unrolled_df['distance'] >= lower_bound) & 
        (unrolled_df['distance'] <= upper_bound) &
        (unrolled_df['id_start'] != reference_id)  
    ]['id_start'].unique()  
    return sorted(filtered_ids.tolist())

reference_id = 1001400 
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result_ids)

def calculate_toll_rate(df)->pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rate_coefficients.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * rate

    return unrolled_df
toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)


def calculate_time_based_toll_rates(df)->pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    weekday_discounts = {
        ('00:00:00', '10:00:00'): 0.8,
        ('10:00:01', '18:00:00'): 1.2,
        ('18:00:01', '23:59:59'): 0.8
    }
    weekend_discount = 0.7
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']
    new_data = []
    for index, row in toll_rate_df.iterrows():
        for day in weekdays + weekends:
            if day in weekdays:
                for time_range, discount in weekday_discounts.items():
                    new_row = row.copy()
                    new_row['start_day'] = day
                    new_row['end_day'] = day
                    new_row['start_time'] = time_range[0]
                    new_row['end_time'] = time_range[1]
                    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                        new_row[vehicle] = row[vehicle] * discount
                    new_data.append(new_row)
            else:
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = '12:00:00'
                new_row['end_time'] = '23:59:59'
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    new_row[vehicle] = row[vehicle] * weekend_discount
                new_data.append(new_row)
    time_based_df = pd.DataFrame(new_data)
    return time_based_df
time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_rate_df)
