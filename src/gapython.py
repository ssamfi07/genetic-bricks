import pandas as pd
import numpy as np
import ast
import random
import concurrent.futures



POPULATION = 10
ITERATIONS = 10

shipping_ranges = './input/ShippingRates.csv'

db = pd.read_csv('csv_exports/database.csv', delimiter=';', dtype=str)
db['Qty'] = db['Qty'].astype(int)
db['Weight'] = db['Weight'].astype(float)
db['store'] = db['store'].apply(ast.literal_eval)
# db['store'] = db['store'].apply(lambda x: pd.Series(x) if isinstance(x, list) else pd.Series([]))



# the last columns into np.arrays
db.Status = db.Status.apply(lambda x: np.fromstring(x[1:-1], sep=',', dtype = int))
db['Stock'] = db['Stock'].apply(lambda x: np.fromstring(x[1:-1], sep=',', dtype = int))
db['Price'] = db['Price'].apply(lambda x: np.fromstring(x[1:-1], sep=',', dtype = float))
db.country = db.country.apply(lambda x: np.array(ast.literal_eval(x)))
db.minvalor = db.minvalor.apply(lambda x: np.fromstring(x[1:-1], sep=',', dtype = float))
db.free = db.free.apply(lambda x: np.fromstring(x[1:-1], sep=',', dtype = float))
db.racio = db.racio.apply(lambda x: np.fromstring(x[1:-1], sep=',', dtype = float))
db.store = db.store.apply(lambda x: np.array(x))

# Create an array of arrays with integers, it represents a single solution
lengths = db['Qty']



# Calculate fitness
def FitnessSolution(solution, stores):
    # Sum the values of 'shipping_cost' and 'bill'
    total_cost = stores['shipping_cost'].sum() + stores['bill'].sum()
    # Penalization

    # minvalue not attended
    count_of_zeros = stores['sol_OK'].value_counts().get(0, 0)

    # print("Count of 0 in 'sol_OK':", count_of_zeros)
    return total_cost + count_of_zeros**2



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
        # pop[i] = single_solution()
        pop[i] , store_solution= second_creation()
    return pop

# cost based only on the price
def item_cost(item, offer, qty):

    return db.loc[item, 'Price'][offer] * qty

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


def CostStores(filtered_solution):
    stores = pd.DataFrame(columns=['stores', 'bill', 'weights', 'country', 'free', 'minvalor'])
    # filtered_solution = np.array([[x for x in arr if x >= 0] for arr in solution], dtype=object)

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


def second_creation():
    solution = np.array([np.arange(length) for length in db.Qty], dtype=object)
    prices = np.zeros((len(db), max(db['Qty'])), dtype=float)
    store_solution = []
    indexes = np.arange(len(db['Qty']))
    np.random.shuffle(indexes)

    for current_index in indexes:
        current_qty = db.loc[current_index, 'Qty']
        store_n = 0
        for i in range(len(solution[current_index])):
            solution[current_index][i] = -1
            # prices[current_index][i] = -1
        
        # print('--------------------Start Index '+ str(current_index)+ '--------------')

        while current_qty > 0:

            for st in store_solution:
                # we search for a store that we are already using but not for the same item
                if st in db.loc[current_index,'store']:
                    if np.any(~np.isin(np.where(db.loc[current_index, 'store'] == st), solution[current_index])):
                        # print('-<_>-')
                        # print(store_solution)
                        for i in random.choice(np.where(db.loc[current_index, 'store'] == st)):
                            if not any(i == x for x in solution[current_index]):
                                # print('Match on: '+ str(st) + ' on the index '+ str(vendor_index))
                                # print(str(i) + ' not in '+ str(solution[current_index]))
                                vendor_index = i
                                solution[current_index][store_n] = vendor_index
                                prices[current_index][store_n] = item_cost(current_index, vendor_index, min(current_qty, db.loc[current_index, 'Stock'][vendor_index]))
                                store_n += 1
                                current_qty -= db.loc[current_index, 'Stock'][vendor_index]
                                break
                    if current_qty <= 0: break
            if current_qty <= 0: break
            # print('# No match: ')
            for i in np.argsort(db.loc[current_index, 'Price']):
                if not any(i == x for x in solution[current_index]):
                    # print(str(i) + ' not in '+ str(solution[current_index]))
                    vendor_index = np.argsort(db.loc[current_index, 'Price'])[i]
                    # vendor_index = random.choice(np.arange(len(db.loc[current_index, 'store'])))
                    solution[current_index][store_n] = vendor_index
                    prices[current_index][store_n] = item_cost(current_index, vendor_index, min(current_qty, db.loc[current_index, 'Stock'][vendor_index]))
                    store_n += 1
                    # print('vendor index = ' + str(vendor_index) + ', Lunghezza = ' + str(db.loc[current_index, 'Stock'][vendor_index]))
                    current_qty -= min(db.loc[current_index, 'Qty'], db.loc[current_index, 'Stock'][vendor_index])
                    # print(db.loc[current_index, 'store'])
                    store_solution.append(db.loc[current_index, 'store'][vendor_index])
                    # print(str(vendor_index)+ ' -> ' + str(db.loc[current_index, 'store'][vendor_index]) )
                    break
    
    filtered_solution = np.array([[x for x in arr if x >= 0] for arr in solution], dtype=object)
    # filtered_prices = np.array([[x for x in arr if x > 0] for arr in prices], dtype=object)
    # print(filtered_prices)
    return filtered_solution, np.array(store_solution), prices

filtered_solution, stores_solution, prices = second_creation()


sr = pd.read_csv(shipping_ranges,delimiter=';')
sr['LRange']= sr.Range.apply(lambda x: x.split(' - ')[0])
sr['URange']= sr.Range.apply(lambda x: x.split(' - ')[1])
sr = sr.drop(columns=['Range']).astype(float)


cost_solution = np.sum(prices)

for st in stores_solution:

    weight = 0
    country = ''
    bill = 0
    for item in range(len(filtered_solution)):
        for batch in range(len(filtered_solution[item])):
                if db.loc[item, 'store'][filtered_solution[item][batch]] == st:
                    weight += db.loc[item, 'Weight'] * min(db.loc[item, 'Weight'],db.loc[item, 'Stock'][filtered_solution[item][batch]])
                    country = db.loc[item, 'country'][filtered_solution[item][batch]]
                    bill += prices[item][batch]

    #we control the shipping costs considering the eventuality that it is free
    # print( db.loc[item, 'free'][filtered_solution[item][batch]])            
    if db.loc[item, 'free'][filtered_solution[item][batch]] == 0 or db.loc[item, 'free'][filtered_solution[item][batch]] > bill:
        # print( db.loc[item, 'free'][filtered_solution[item][batch]])
        if country not in sr.columns:
            cost_solution += sr.loc[(sr['LRange'] < weight) & (sr['URange']> weight), 'All Other'].values[0]
        else:
            cost_solution += sr.loc[(sr['LRange'] < weight) & (sr['URange']> weight), country].values[0]
    else: 
        print('Avoinded shipping costs of ' + str(db.loc[item, 'free'][filtered_solution[item][batch]]) + ' Bill was ' + str(bill))

print (cost_solution)

    


          


    




