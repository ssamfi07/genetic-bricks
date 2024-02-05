import pandas as pd
import numpy as np
import ast
import random
import concurrent.futures



POPULATION = 100
ITERATIONS = 250
CLONING_RATE = 0.05

shipping_ranges = './input/ShippingRates.csv'
vendors_csv = './input/vendors.csv'

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
    stores = np.full((len(db), max(db['Qty'])), '',dtype='U100')
    prices = np.zeros((len(db), max(db['Qty'])), dtype=float)

    # Repeat the inner array 500 times to create the outer array
    pop = np.array([sol.copy() for _ in range(POPULATION)], dtype=object)
    sts = np.array([stores.copy() for _ in range(POPULATION)], dtype=object)
    pcs = np.array([prices.copy() for _ in range(POPULATION)], dtype=object)

    # Print the shape of the outer array
    for i in range(POPULATION):
        # pop[i] = single_solution()
        pop[i] , sts[i], pcs[i]= second_creation()
    return pop, sts, pcs

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
    store_solution = np.full((len(db), max(db['Qty'])), '',dtype='U100')
    # print(store_solution)
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

            for st in np.unique(store_solution.flatten()):
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
                # print(i)
                if not any(i == x for x in solution[current_index]):
                    vendor_index = np.argsort(db.loc[current_index, 'Price'])[i]
                    # print('Item '+ str(db.loc[current_index, 'Item No'])+ ': '+str(store_n) + ' -> ' + str(db.loc[current_index, 'Price'][vendor_index]))
                    solution[current_index][store_n] = vendor_index
                    prices[current_index][store_n] = item_cost(current_index, vendor_index, min(current_qty, db.loc[current_index, 'Stock'][vendor_index]))
                    store_solution[current_index][store_n] = db.loc[current_index, 'store'][vendor_index]

                    store_n += 1
                    current_qty -= min(db.loc[current_index, 'Qty'], db.loc[current_index, 'Stock'][vendor_index])
                    break
    
    filtered_solution = np.array([[x for x in arr if x >= 0] for arr in solution], dtype=object)
    # filtered_prices = np.array([[x for x in arr if x > 0] for arr in prices], dtype=object)
    # print(filtered_prices)
    return filtered_solution, store_solution, prices


def FitnessOK(filtered_solution, stores_solution, prices):

    stores_solution= np.unique(stores_solution[stores_solution != '']. flatten())
    sr = pd.read_csv(shipping_ranges,delimiter=';')
    sr['LRange']= sr.Range.apply(lambda x: x.split(' - ')[0])
    sr['URange']= sr.Range.apply(lambda x: x.split(' - ')[1])
    sr = sr.drop(columns=['Range']).astype(float)

    cost_solution = 0
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

        # we control the shipping costs considering the eventuality that it is free
        # print( db.loc[item, 'free'][filtered_solution[item][batch]])            
        if db.loc[item, 'free'][filtered_solution[item][batch]] == 0 or db.loc[item, 'free'][filtered_solution[item][batch]] > bill:
            # print( db.loc[item, 'free'][filtered_solution[item][batch]])
            if country not in sr.columns:
                # print(weight)
                cost_solution += sr.loc[(sr['LRange'] <= weight) & (sr['URange']> weight), 'All Other'].values[0]
            else:
                cost_solution += sr.loc[(sr['LRange'] <= weight) & (sr['URange']> weight), country].values[0]
                # print('ok')
        # else: 
        #     print('Avoinded shipping costs of ' + str(db.loc[item, 'free'][filtered_solution[item][batch]]) + ' Bill was ' + str(bill))
    return cost_solution

def GenerateChild(filtered_solution_1, store_solution_1, prices_1, filtered_solution_2, store_solution_2, prices_2):
# filtered_solution_1, store_solution_1, prices_1 = second_creation()
# filtered_solution_2, store_solution_2, prices_2 = second_creation()

    child_solution = np.array([np.arange(length) for length in db.Qty], dtype=object)
    child_stores = np.full((len(db), max(db['Qty'])), '',dtype='U100')
    child_prices = np.zeros((len(db), max(db['Qty'])), dtype=float)
    child_qtys = np.array(db['Qty'])

    indexes = np.arange(len(db['Qty']))
    np.random.shuffle(indexes)

    for i in indexes:
        child_solution[i] = np.full_like(child_solution[i], -1)
        batch = 0
        for inter in np.random.permutation(np.intersect1d(filtered_solution_1[i], filtered_solution_2[i])):
            child_solution[i][batch] = inter
            child_stores[i][batch] = db.loc[i, 'store'][inter]
            child_prices[i][batch] += item_cost(i, inter, min(child_qtys[i], db.loc[i, 'Stock'][inter]))
            child_qtys[i] -= min(child_qtys[i], db.loc[i, 'Stock'][inter])
            batch += 1
            if child_qtys[i] <= 0: break
    np.random.shuffle(indexes)
    np.unique(child_stores[child_stores != '']. flatten())
    for i in indexes:
        # np.unique(child_stores[child_stores != '']. flatten())
        # print('Enter index ' + str(i))
        while child_qtys[i] > 0:
            
            batch = np.count_nonzero(child_solution[i] > 0)
            # print('The cs is '+ str(child_solution[i]) + ' so batch is ' + str(batch))
            for j in range(len(filtered_solution_1[i])):
                if store_solution_1[i][j] in np.unique(child_stores[child_stores != ''].flatten()):
                    # print('match on '+ store_solution_1[i][j])
                    child_solution[i][batch] = filtered_solution_1[i][j]
                    child_prices[i][batch] = item_cost(i, filtered_solution_1[i][j], min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_1[i][j]]))
                    child_stores[i][batch] = store_solution_1[i][j]
                    child_qtys[i] -= min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_1[i][j]])
                    batch += 1
                    if child_qtys[i] <= 0: break 
            if child_qtys[i] <= 0: break

            for j in range(len(filtered_solution_2[i])):
                if store_solution_2[i][j] in np.unique(child_stores[child_stores != ''].flatten()):
                    # print('match on '+ store_solution_2[i][j])                  
                    child_solution[i][batch] = filtered_solution_2[i][j]
                    child_prices[i][batch] = item_cost(i, filtered_solution_2[i][j], min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_2[i][j]]))
                    child_stores[i][batch] = store_solution_2[i][j]
                    child_qtys[i] -= min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_2[i][j]])
                    batch += 1
                    if child_qtys[i] <= 0: break 
            if child_qtys[i] <= 0: break


            # print('we  go to the randoms, ' + str(child_qtys[i]))

            selected = -1
            if random.choice([1, 2]) == 1:
                for j in np.argsort(db.loc[i, 'Price'][filtered_solution_1[i]]):
                    if db.loc[i, 'Price'][filtered_solution_1[i][j]] not in child_solution[i]:
                        # print('found')
                        child_solution[i][batch] = filtered_solution_1[i][j]
                        child_prices[i][batch] = item_cost(i, filtered_solution_1[i][j], min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_1[i][j]]))
                        # min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_1[i][j]])
                        child_stores[i][batch] = store_solution_1[i][j]
                        child_qtys[i] -= min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_1[i][j]])
                        batch += 1
                        if child_qtys[i] <= 0: break
                        
            else:
                for j in np.argsort(db.loc[i, 'Price'][filtered_solution_2[i]]):
                    if db.loc[i, 'Price'][filtered_solution_2[i][j]] not in child_solution[i]:
                        # print('found')
                        child_solution[i][batch] = filtered_solution_2[i][j]
                        child_prices[i][batch] = item_cost(i, filtered_solution_2[i][j], min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_2[i][j]]))
                        # min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_1[i][j]])
                        child_stores[i][batch] = store_solution_2[i][j]
                        child_qtys[i] -= min(child_qtys[i], db.loc[i, 'Stock'][filtered_solution_2[i][j]])
                        batch += 1
                        if child_qtys[i] <= 0: break


    filtered_child= np.array([[x for x in arr if x >= 0] for arr in child_solution], dtype=object)

    # print('Parent 1 ->' +   str(np.sum(prices_1)))
    # print(FitnessOK(filtered_solution_1,store_solution_1, prices_1))
    # print(len(np.unique(store_solution_1[store_solution_1 != '']. flatten())))



    # print('Parent 2 ->' +   str(np.sum(prices_2)))
    # print(FitnessOK(filtered_solution_2,store_solution_2, prices_2))
    # print(len(np.unique(store_solution_2[store_solution_2 != '']. flatten())))


    # print
    # print('Child solution ->' +   str(np.sum(child_prices)))
    # print(FitnessOK(filtered_child, child_stores, child_prices))
    # print(len(np.unique(child_stores[child_stores != '']. flatten())))
    return filtered_child, child_stores, child_prices

pop, sts, pcs = initialize_population()

fitnesses = np.arange(POPULATION, dtype = float)
probabilities = np.arange(POPULATION, dtype = float)

for i in range(POPULATION):
    fitnesses[i] = FitnessOK(pop[i], sts[i], pcs[i])
for iter in range(ITERATIONS):

    # Calculate the reciprocal of fitness values
    fitup = 1 / fitnesses

    # Normalize the reciprocals to get probabilities
    selection_probabilities = fitup / np.sum(fitup)

    sol = np.array([np.arange(length) for length in db['Qty']], dtype=object)
    stores = np.full((len(db), max(db['Qty'])), '',dtype='U100')
    prices = np.zeros((len(db), max(db['Qty'])), dtype=float)

    # Repeat the inner array 500 times to create the outer array
    new_pop = np.array([sol.copy() for _ in range(POPULATION)], dtype=object)
    new_sts = np.array([stores.copy() for _ in range(POPULATION)], dtype=object)
    new_pcs = np.array([prices.copy() for _ in range(POPULATION)], dtype=object)

    cloned = 0
    sorted_solution = np.argsort(fitup)


    for i in range(POPULATION):
        if random.random() < CLONING_RATE:
            # print("Let's clone")
            new_pop[i], new_sts[i], new_pcs[i] = pop[sorted_solution[cloned]], sts[sorted_solution[cloned]], pcs[sorted_solution[cloned]]
        else:
            roulette_spin = random.random()
            parent_1 = np.random.choice(np.arange(POPULATION), p=selection_probabilities)
            parent_2 = parent_1
            while parent_2 == parent_1:
                parent_2 = np.random.choice(np.arange(POPULATION), p=selection_probabilities)
            
            new_pop[i], new_sts[i], new_pcs[i] = GenerateChild(pop[parent_1], sts[parent_1], pcs[parent_1], pop[parent_2], sts[parent_1], pcs[parent_2])


    new_fitnesses = np.arange(POPULATION, dtype = float)
    for i in range(POPULATION):
        new_fitnesses[i] = FitnessOK(new_pop[i], new_sts[i], new_pcs[i])
        # print(new_fitnesses[i])

    pop, sts, pcs = new_pop, new_sts, new_pcs
    fitnesses = new_fitnesses
    print('finishing ' + str(iter))
    

best = np.argmin(fitnesses)
print(pop[best])
print(fitnesses[best])
print(np.unique(sts[best].flatten()))


# filtered_solution, store_solution, prices = second_creation()
# best_cost = FitnessOK(filtered_solution, store_solution, prices)
# print('Best cost = '+ str(best_cost))

# stores = np.unique(store_solution[store_solution != ''])
# dbstore = pd.read_csv(vendors_csv,delimiter=';')[['store', 'country']]
# sr = ShippingRanges(shipping_ranges)
# countries= np.array(['']* len(stores),dtype='U100')

# for i in range(len(stores)): 

#     countries[i]= dbstore.loc[dbstore['store']==stores[i], 'country'].values[0]


# for st in np.random.permutation(stores):
#     ls_solution = np.array([np.arange(length) for length in db.Qty], dtype=object)
#     ls_prices = np.zeros((len(db), max(db['Qty'])), dtype=float)
#     ls_store = np.full((len(db), max(db['Qty'])), '',dtype='U100')

#     print('Analyze '+ st)
#     for item in range(len(filtered_solution)):
#         qty = db.loc[item,'Qty']
#         store_n = 0

#         for i in range(len(ls_solution[item])):
#             ls_solution[item][i] = -1
        
#         if (st in db.loc[item, 'store']):
            
#             for j in np.argsort(db.loc[item, 'Price'][db.loc[item, 'store'] == st]):
#                 if (db.loc[item, 'store'][j]== st) and (j not in filtered_solution[item]) and (db.loc[item, 'store'][j] == st):
#                     qty -= min(qty, db.loc[item, 'Stock'][db.loc[item, 'store'] == st][j])
#                     ls_solution[item][store_n] = np.arange(len(db.loc[item, 'Stock']))[db.loc[item, 'store'] == st][j]
#                     ls_prices = item_cost(item, j, min(qty, db.loc[item, 'Stock'][db.loc[item, 'store'] == st][j]))
#                     ls_store[item][store_n] = st
#                     store_n += 1

#                     if qty <= 0: break
#                     # np.arange(len(db.loc[item, 'Stock']))[db.loc[item, 'store'] == st][j]
#                     # db.loc[item, 'Stock'][db.loc[item, 'store'] == st][j]
#         while qty > 0:
#             for j in np.argsort(db.loc[item, 'Price']):
#                 if j not in ls_solution[item] and [db.loc[item, 'store'][j] in stores]:
#                         qty -= min(qty, db.loc[item, 'Stock'][j])
#                         ls_solution[item][store_n] = j
#                         ls_prices = item_cost(item, j, min(qty, db.loc[item, 'Stock'][j]))
#                         ls_store[item][store_n] = st
#                         store_n += 1
#                         if qty <= 0: break

#     print(ls_prices)
#     print(FitnessOK(ls_solution, ls_store, ls_prices))

    # FitnessOK(filtered_solution, stores_solution, prices)



























                    
                    

    
 



                    


    











