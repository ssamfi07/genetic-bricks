import pandas as pd  # Assuming you'll use pandas for data manipulation
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

    # # merge instance_parts with itemcostvendor on Item No and Color ID
    instance_parts_vendors = pd.merge(instance_parts, items, on=['Item No', 'Color ID'], how='left')

    # # Create a new column 'Variants' containing lists of entries for each unique pair of 'Item No' and 'Color ID'
    # instance_parts_vendors['Variants'] = instance_parts_vendors.groupby(['Item No', 'Color ID']).apply(lambda group: group[
    #     ['Status', 'Stock', 'store', 'currency', 'unitprice']].values.tolist()).reset_index(drop=True)
    # print(instance_parts_vendors)

    # # merge instance_parts_vendors with exchange rates on currency
    # instance_parts_vendors_exchange = pd.merge(instance_parts_vendors, exchange_rates, on='currency')
    # # simple_export(instance_parts_vendors_exchange, 'wtf.csv')

    # # merge instance_parts_vendors with colors on color id
    # instance_parts_vendors_exchange.drop(['Color ID_y'], axis=1, inplace=True)
    # instance_parts_vendors_exchange.rename(columns={'Color ID_x': 'Color ID'}, inplace=True)
    # instance_parts_vendors_exchange_colors = pd.merge(instance_parts_vendors_exchange, colors, on='Color ID')

    # # merge instance_parts_vendors_exchange_colors with vendors on store
    # instance_parts_vendors_exchange_colors = pd.merge(instance_parts_vendors_exchange_colors, vendors, on='store')
    
    # #filter out stupid columns
    # instance_parts_vendors_exchange_colors.drop(['Number', 'Description', 'Color Name', 'RGB', 'Type', 'Parts', 'In Sets', 'Wanted', 'For Sale', 'Year From', 'Year To'], axis=1, inplace=True)
    simple_export(instance_parts_vendors, 'wtf.csv')

create_data_structure()
