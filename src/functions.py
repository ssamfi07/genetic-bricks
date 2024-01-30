import pandas as pd
import numpy as np
import ast
import random

# Calculate fitness
def FitnessSolution(solution, stores):
    # Sum the values of 'shipping_cost' and 'bill'
    total_cost = stores['shipping_cost'].sum() + stores['bill'].sum()
    # Penalization

    # minvalue not attended
    count_of_zeros = stores['sol_OK'].value_counts().get(0, 0)

    # print("Count of 0 in 'sol_OK':", count_of_zeros)
    return total_cost + count_of_zeros**2

def CostStores(solution):
    stores = pd.DataFrame(columns=['stores', 'bill', 'weights', 'country', 'free', 'minvalor'])
    filtered_solution = np.array([[x for x in arr if x >= 0] for arr in solution], dtype=object)

    for i in range(len(filtered_solution)):
        for j in range(len(filtered_solution[i])):
            store_name = db.loc[i, 'store'][filtered_solution[i][j]]
            if store_name not in stores.stores:
                stores = stores.append({
                    'stores': store_name, 
                    'bill' : item_cost(i, filtered_solution[i][j]),
                    'weights': db.loc[i, 'Weight'],
                    'country': db.loc[i, 'country'][filtered_solution[i][j]],
                    'free': db.loc[i, 'free'][filtered_solution[i][j]],
                    'minvalor' : db.loc[i, 'minvalor'][filtered_solution[i][j]]
                    }, ignore_index= True)
                
            else:
                stores.loc[stores.stores == store_name, 'weight'] += db.loc[i, 'Weight']
                stores.loc[stores.stores == store_name, 'bill'] += item_cost(i, filtered_solution[i][j])

    stores['shipping_cost'] = stores.apply(ShippingCost, axis= 1)
    stores['shipping_cost'] = stores.apply(FreeShipping, axis= 1)
    stores['sol_OK'] = stores.apply(Admissible_Solution, axis=1)


    stores = stores.drop(columns= [
        'free', 
        'country', 
        'weights',
        'minvalor'
        ])


    return stores

def TotalCost(solution, shipping_ranges):
    return item_cost(solution) + ShippingCost(solution, ShippingRanges(shipping_ranges)) 

def Admissible_Solution (serie): 
    if serie['minvalor'] > serie['bill']:
        # this is not Admissable
        return 0
    else:
        return 1

def FreeShipping(serie):
    if serie['free'] < serie['bill'] and serie['free'] > 0:
        return 0
    else:
        return serie['shipping_cost']
    
def ShippingCost(serie):
    sr = ShippingRanges(shipping_ranges)
    if(sr.loc[0, 'Range'] > serie['weights']):
        if(serie['country'] in sr.columns):
            return sr.loc[0, serie['country']]  
        else:
            return sr.loc[0, 'All Other']

    for i in range(len(sr)-1):
        if(sr.loc[i, 'Range'] < serie['weights'] and sr.loc[i + 1, 'Range'] > serie['weights']):
            if(serie['country'] in sr.columns):
                return sr.loc[i, serie['country']]  
            else:
                return sr.loc[i, 'All Other']

# read shipping weight ranges
def ShippingRanges(shipping_ranges):
    sr = pd.read_csv(shipping_ranges, delimiter=';')
    sr.Range= sr.Range.apply(
            lambda x: 
                int(x.split('-')[1].strip())     
            )
    sr = sr.astype(float)
    return sr

# create population
def initialize_population():
    # Create the inner array of arrays
    sol = np.array([np.arange(length) for length in db['Qty']], dtype=object)
    # Repeat the inner array 500 times to create the outer array
    pop = np.array([sol.copy() for _ in range(POPULATION)], dtype=object)
    # Print the shape of the outer array
    for i in range(POPULATION):
        pop[i] = single_solution()
    return pop

# cost based only on the price
def item_cost(item, price):

    return db.loc[item, 'Price'][price] * min(db.loc[item, 'Stock'][price], db.loc[item, 'Qty'])

# generate a compleately random solution
def single_solution():
    solution = np.array([np.arange(length) for length in db.Qty], dtype=object)
    for index in range(len(db.index)):
        index_qty = db.iloc[index, 2]
        while index_qty > 0:
            solution[index] = np.full_like(solution[index], -1)
            item_iter = 0
            while True:
                random_number = random.randint(0, len(db.iloc[index, 6])-1)
                if not np.isin(random_number, solution[index]):
                    solution[index][item_iter] = random_number
                    index_qty -= min(index_qty, db.iloc[index, 6][random_number])
                    break
            item_iter += 1
    return solution