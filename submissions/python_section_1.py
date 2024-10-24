from typing import Dict, List
import pandas as pd
import re
import math
import polyline


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    reversed_list = []
    for i in range(0, len(lst), n):
        end = min(i + n, len(lst))
        for j in range(end - 1, i - 1, -1):
            reversed_list.append(lst[j])
    return reversed_list
print(reverse_by_n_elements([1,2,3,4,5,6,7,8],3))

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for i in lst:
        length = len(i)
        if length not in length_dict:
            length_dict[length] = []  
        length_dict[length].append(i)  

    sorted_length_dict = {}
    for i in range(1, max(length_dict.keys()) + 1):
        if i in length_dict:
            sorted_length_dict[i] = length_dict[i]
    return sorted_length_dict
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))  
   

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}

    def flatten(current_dict, parent_key: str = ''):
        if isinstance(current_dict, dict):
            for l, n in current_dict.items():
                new_key = f"{parent_key}{sep}{l}" if parent_key else l
                flatten(n, new_key)
        elif isinstance(current_dict, list):
            for i, j in enumerate(current_dict):
                new_key = f"{parent_key}[{i}]"
                flatten(j, new_key)
        else:
            flattened[parent_key] = current_dict
    flatten(nested_dict)
    return flattened
 

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def combination(n):
        if len(n) == len(nums):
            result.append(n[:])
            return
        
        for i in range(len(nums)):
            if element[i] or (i > 0 and nums[i] == nums[i - 1] and not element[i - 1]):
                continue
            
            element[i] = True
            n.append(nums[i])
            combination(n)
            n.pop()
            element[i] = False

    nums.sort()
    result = []
    element = [False] * len(nums)
    combination([])
    return result
input_list = [1, 1, 2]
print(unique_permutations(input_list))

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    valid_dates = re.findall(pattern, text)
    return valid_dates
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))  


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great-circle distance between two points on the Earth
        specified in decimal degrees using the Haversine formula.
        
        Args:
            lat1 (float): Latitude of the first point.
            lon1 (float): Longitude of the first point.
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the second point.
        
        Returns:
            float: Distance between the two points in meters.
        """
      
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        r = 6371000  
        return c * r
    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    distances = [0] 
    for i in range(1, len(coordinates)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)
    
    df['distance'] = distances
    return df
polyline_str = "_p~iF~wuy@_@_@iDof@dAcAf@k@bD|@_@uAlAyAq@d@_AlC]fA" 
df = polyline_to_dataframe(polyline_str)
print(df)

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    final_matrix = []
    
    for i in range(n):
        row_sum = sum(rotated_matrix[i]) 
        final_row = []
        for j in range(n):
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  
            final_value = (row_sum - rotated_matrix[i][j]) + (col_sum - rotated_matrix[i][j])
            final_row.append(final_value)
        final_matrix.append(final_row)
    return final_matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
print(result)

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    results = pd.Series(index=df[['id', 'id_2']].drop_duplicates().set_index(['id', 'id_2']).index, dtype=bool)

    for (id, id_2), group in df.groupby(['id', 'id_2']):
        covered_days = set()
        time_per_day = {day: [] for day in range(7)}

        for index, row in group.iterrows():
            start_day = day_mapping[row['startDay']]
            end_day = day_mapping[row['endDay']]
            start_time, end_time = row['startTime'], row['endTime']
            
            covered_days.update([start_day, end_day])
            time_per_day[start_day].append((start_time, '23:59:59' if start_day != end_day else end_time))
            if start_day != end_day:
                time_per_day[end_day].append(('00:00:00', end_time))

        all_days_covered = len(covered_days) == 7
        full_coverage = all(
            times and (times[0][0] == '00:00:00' and times[-1][1] == '23:59:59')
            for day, times in time_per_day.items() if times
        )
        
        results.loc[(id, id_2)] = not (all_days_covered and full_coverage)

    return results
df=pd.read_csv("datasets\dataset-1.csv")
incomplete_pairs = time_check(df)
print(incomplete_pairs)