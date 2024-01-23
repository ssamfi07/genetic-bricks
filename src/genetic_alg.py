import random
import time

# Data structure for an item
class Item:
    def __init__(self, itemNo, itemName, qty, colorID, categoryID, categoryName, weight, dimensions, status, stock,
                 store, currency, unitPrice, cambio, country, minValor, free, racio):
        self.itemNo = itemNo
        self.itemName = itemName
        self.qty = qty
        self.colorID = colorID
        self.categoryID = categoryID
        self.categoryName = categoryName
        self.weight = weight
        self.dimensions = dimensions
        self.status = status
        self.stock = stock
        self.store = store
        self.currency = currency
        self.unitPrice = unitPrice
        self.cambio = cambio
        self.country = country
        self.minValor = minValor
        self.free = free
        self.racio = racio

# Constants for genetic algorithm
POPULATION_SIZE = 500
NUM_GENERATIONS = 250
CROSSOVER_RATE = 0.75
MUTATION_RATE = 0.01
CLONING_RATE = 0.05

# Random number generator
rng = random.Random(time.time())

def initialize_problem(filename):
    with open(filename, 'r') as input_file:
        items = []
        input_file.readline()  # Skip the header line

        for line in input_file:
            # Parse the line to initialize an Item object
            # (You need to adapt this part based on your file structure)
            # Example:
            # item_data = line.strip().split(';')
            # item = Item(*item_data)
            # items.append(item)

    return items

# Function to calculate the fitness of a solution (to be minimized)
def calculate_total_cost(items):
    total_cost = 0.0
    for item in items:
        total_cost += item.qty * item.unitPrice
    return total_cost

# Function to perform crossover
def crossover(parent1, parent2, crossover_rate):
    child = [None] * len(parent1)

    if rng.random() < crossover_rate:
        crossover_point = rng.randint(0, len(parent1))
        child[:crossover_point] = parent1[:crossover_point]
        child[crossover_point:] = parent2[crossover_point:]
    else:
        child = parent1 if rng.random() < 0.5 else parent2

    return child

# Function to perform mutation
def mutation(solution):
    for item in solution:
        mutation_chance = rng.random()
        if mutation_chance < MUTATION_RATE:
            # Mutate the quantity (Qty) based on the available stock
            item.qty = min(item.qty + 1, item.stock)

# Function to generate a random initial population
def generate_population(population_size, items):
    population = []
    for _ in range(population_size):
        individual = [Item(rng.randint(0, item.stock) for item in items)]
        population.append(individual)

    return population

# Function to run the genetic algorithm
def genetic_algorithm(items):
    start = time.time()
    with open("genetic.txt", 'w') as output_file:
        population = generate_population(POPULATION_SIZE, items)

        for generation in range(NUM_GENERATIONS):
            # Add fitnesses for each individual in the population
            fitnesses = [(calculate_total_cost(individual), i) for i, individual in enumerate(population)]

            # Compute the total fitness of the population
            total_fitness = sum(fitness for fitness, _ in fitnesses)

            # Compute the probabilities of selection for each individual
            selection_probabilities = [fitness / total_fitness for fitness, _ in fitnesses]

            # Create a new population
            new_population = []

            # Generate random children by crossover
            for _ in range(POPULATION_SIZE):
                # Try to clone
                if rng.random() < CLONING_RATE:
                    new_population.append(population[fitnesses[i][1]])
                else:
                    # Spin the roulette wheel to select the first parent
                    roulette_spin = rng.random()
                    cumulative_probability = 0.0
                    parent1_index = next(i for i, prob in enumerate(selection_probabilities) if (cumulative_probability := cumulative_probability + prob) >= roulette_spin)

                    # Spin the roulette wheel to select the second parent
                    roulette_spin = rng.random()
                    cumulative_probability = 0.0
                    parent2_index = next(i for i, prob in enumerate(selection_probabilities) if (cumulative_probability := cumulative_probability + prob) >= roulette_spin)

                    # Perform crossover and mutation to create the child
                    child = crossover(population[parent1_index], population[parent2_index], CROSSOVER_RATE)
                    mutation(child)
                    new_population.append(child)

            # Elitism: Always keep the best 5 solutions for each new generation
            new_population[:5] = population[:5]

            # Sort population by fitness in descending order
            fitnesses.sort(reverse=True)
            population = [new_population[i] for _, i in fitnesses]

            end = time.time()
            time_elapsed = end - start

            # Print best individual fitnesses and time (converted to seconds)
            output_file.write(f"{fitnesses[0][0]} {time_elapsed}\n")

    return population[0]

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "<input_file>")
        sys.exit(1)

    items = initialize_problem(sys.argv[1])

    # Run genetic algorithm and print solution
    genetic_solution = genetic_algorithm(items)

if __name__ == "__main__":
    main()
