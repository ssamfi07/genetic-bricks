import pandas as pd  # Assuming you'll use pandas for data manipulation
import numpy as np
import os

# Function to read data from a file and extract relevant fields
def read_data(file_path, relevant_fields, delimiter = '\t'):
    data = pd.read_csv(file_path, delimiter=delimiter)
    return data[relevant_fields]

# Function to calculate the 'price' field
def calculate_price(row):
    return row['unitprice'] * row['rate']

# Function to determine 'sndvalor' based on 'Qty', 'Ranges x Countries', and 'Weight'
# def determine_sndvalor(row):
    # Implement your logic here
    # You may need to access other data files (e.g., ShippingRates) for weight ranges and their associated costs
    # Use the provided logic for 'sndvalor' determination based on 'Qty', 'Ranges x Countries', and 'Weight'
    # return calculated_sndvalor

def simple_export(df, fileName):
    # get the current working directory
    current_dir = os.getcwd()
    # define the path to the csv_exports folder
    exports_folder = os.path.join(current_dir, '..', 'csv_exports')
    # if the folder doesn't exist, create it
    if not os.path.exists(exports_folder):
        os.makedirs(exports_folder)
    # define the full path to the CSV file
    full_path = os.path.join(exports_folder, fileName)
    df.to_csv(full_path, sep=';', index=False, encoding='utf-8')

# Function to calculate volume
def calculate_volume(dimensions):
    try:
        # Split the dimensions and convert to integers
        parts = [int(part) if part.isdigit() else 0 for part in dimensions.split(' x ')]
        # Calculate volume
        volume = parts[0] * parts[1] * parts[2]
        return volume
    except (ValueError, IndexError):
        return 0
    
def count_multiple_store_values(df):
    # Count the number of unique store values for each group
    unique_stores = df.groupby(['Item No', 'Color ID', 'Qty', 'Weight', 'Volume'])['store'].nunique()

    # Create a new column indicating whether there are multiple unique store values
    multiple_stores = unique_stores > 1

    # Return the DataFrame with the new column
    return df.assign(multiple_stores=multiple_stores)



# Main function to create the data structure
def create_data_structure():
    # item fields
    item_fields = ['Item No', 'Color ID', 'Description', 'Status', 'Stock', 'store', 'unitcurrency', 'unitprice']
    # instance fields
    instance_fields = ['Item No', 'Item Name', 'Qty', 'Color ID']
    # exchange rate fields
    exchange_rate_fields = ['sndmoeda', 'cambio']
    # parts fields
    parts_fields = ['Category ID', 'Category Name', 'Number', 'Name', 'Weight (in Grams)', 'Dimensions']
    # colors fields
    colors_fields = ['Color ID', 'Color Name', 'RGB', 'Type', 'Parts', 'In Sets', 'Wanted', 'For Sale', 'Year From', 'Year To']
    # vendros fields
    vendors_fields = ['store', 'country', 'minvalor', 'free', 'racio']

    # read data for the instance
    instance = read_data('../input/Instances/S-9500-1.txt', instance_fields)
    # read data for the items
    items = read_data('../input/itemcostvendor.csv', item_fields)
    # read the data for exchange rates
    exchange_rates = read_data('../input/euroexchangerate.csv', exchange_rate_fields, delimiter=';')
    # read the data for parts
    parts = read_data('../input/Parts.txt', parts_fields)
    # read the data for colors
    colors = read_data('../input/colors.txt', colors_fields)
    # read the data for vendors
    vendors = read_data('../input/vendors.csv', vendors_fields, delimiter=';')

    # rename the Name from parts to Item Name for consistency
    parts.rename(columns={'Name': 'Item_Name'}, inplace=True)
    instance.rename(columns={'Item Name': 'Item_Name'}, inplace=True)

    # rename the Name from parts to Item Name for consistency
    exchange_rates.rename(columns={'sndmoeda': 'currency'}, inplace=True)
    items.rename(columns={'unitcurrency': 'currency'}, inplace=True)

    # Calculate the 'price' field
    # item_cost_data['price'] = item_cost_data.apply(calculate_price, axis=1)

    # Determine 'sndvalor' field
    # item_cost_data['sndvalor'] = item_cost_data.apply(determine_sndvalor, axis=1)

    # Merge the dataframes on the 'Item Name' field
    instance_parts = pd.merge(instance, parts, on='Item_Name')
    instance_parts.drop(['Number', 'Category ID', 'Category Name'], axis=1, inplace=True)
    instance_parts.rename(columns={'Weight (in Grams)': 'Weight'}, inplace=True)
    instance_parts['Volume'] = instance_parts['Dimensions'].apply(calculate_volume)
    instance_parts.drop(['Dimensions'], axis=1, inplace=True)

    # merge instance_parts with itemcostvendor on Item No and Color ID
    instance_parts_vendors = pd.merge(instance_parts, items, on=['Item No', 'Color ID'], how='left')

    exchange_rates_dict = dict(zip(exchange_rates['currency'], exchange_rates['cambio']))
    # vendors_dict = dict(zip(vendors['store'], vendors['country'], vendors['minvalor'], vendors['free'], vendors['racio']))

    def calculate_price(row):
        try:
            unitprice = float(row['unitprice'])
        except ValueError:
            # If conversion to float fails, drop the row
            return pd.NA
        currency = row['currency']
        exchange_rate = exchange_rates_dict.get(currency, 1)  # Default to 1 if currency not found
        price = unitprice * exchange_rate
        return round(price, 3)

    # Define a function to encode 'Used' as 0 and 'New' as 1
    def encode_status(status_list):
        return [0 if status == 'Used' else 1 for status in status_list]

    # Function to process stock values
    def encode_stock(stock_list):
        return [process_stock_value(value) for value in stock_list]

    # Function to remove quotes and convert to integer
    def process_stock_value(value):
        try:
            # Attempt to convert to integer without quotes
            return int(value)
        except ValueError:
            # Handle the case where conversion to integer fails
            print(f"Warning: Unable to convert '{value}' to integer.")
            return None  # or any other appropriate handling

    instance_parts_vendors = pd.merge(instance_parts_vendors, vendors, on='store', how='left')

    # Apply the function to create a new column 'Price'
    # print(instance_parts_vendors['unitprice'])
    instance_parts_vendors['Price'] = instance_parts_vendors.apply(calculate_price, axis=1)

    instance_parts_vendors['country'] = instance_parts_vendors['country'].fillna('Unknown')  # Replace NaN with 'Unknown' or any default value
    instance_parts_vendors['minvalor'] = instance_parts_vendors['minvalor'].fillna(0.0)  # Replace NaN with 0 or any default value
    instance_parts_vendors['free'] = instance_parts_vendors['free'].fillna(0.0)  # Replace NaN with 0 or any default value
    instance_parts_vendors['racio'] = instance_parts_vendors['racio'].fillna(0.0)  # Replace NaN with 0.0 or any default value

    grouped_df = instance_parts_vendors.groupby(['Item No', 'Color ID', 'Qty', 'Weight', 'Volume']).agg({
        # 'Description': list,
        'Status': list,
        'Stock': list,
        'Price': list,
        'country': list,
        'minvalor': list,
        'free': list,
        'racio': list,
        'store': list,
    }).reset_index()

    # # Apply the encoding function to the 'Status' column
    # grouped_df['Status'] = grouped_df['Status'].apply(encode_status)

    # # Apply the encoding function to the 'Stock' column
    # grouped_df['Stock'] = grouped_df['Stock'].apply(encode_stock)


    # mask = grouped_df['Stock'].apply(lambda x: pd.isna(x[0]) if len(x) > 0 else False)
    # masked_df = grouped_df[~mask]




# # Apply the function to the grouped_df
#     df = count_multiple_store_values(grouped_df.copy())
#     print(df)
#     # export final dataframe
#     # simple_export(masked_df, 'database.csv')

create_data_structure()
