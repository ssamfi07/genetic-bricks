#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iterator>

using namespace std;

// Data structure for an item
struct Item
{
    string itemNo;
    string itemName;
    int qty;
    string colorID;
    string categoryID;
    string categoryName;
    int weight;
    string dimensions;
    string status;
    int stock;
    string store;
    string currency;
    double unitPrice;
    double cambio;
    string country;
    double minValor;
    bool free;
    double racio;
};

// Constants for genetic algorithm
const int POPULATION_SIZE = 500;
const int NUM_GENERATIONS = 250;
const double CROSSOVER_RATE = 0.75;
const double MUTATION_RATE = 0.01;
const double CLONING_RATE = 0.05;

// Random number generator
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

vector<Item> initialize_problem(string filename)
{
    ifstream input_file(filename);
    if (!input_file.is_open())
    {
        cerr << "Error: Unable to open input file." << endl;
        exit(1);
    }

    vector<Item> items;
    string line;

    // Skip the header line
    getline(input_file, line);

    while (getline(input_file, line))
    {
        // the first line has general info of the item

        // the next lines represent all the possible variants an item can have, based on the stores
        
    }

    input_file.close();
    return items;
}


// Function to calculate the fitness of a solution (to be minimized)
double calculate_total_cost(const vector<Item>& items)
{
    double total_cost = 0.0;

    for (const Item& item : items)
    {
        total_cost += item.qty * item.unitPrice;
    }

    return total_cost;
}

// Function to perform crossover
vector<Item> crossover(const vector<Item>& parent1, const vector<Item>& parent2, double crossover_rate)
{
    vector<Item> child(parent1.size());

    // generate a random double between 0 and 1
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (dist(rng) < crossover_rate)
    {
        int crossover_point = dist(rng) * parent1.size();
        for (int i = 0; i < crossover_point; i++)
        {
            child[i] = parent1[i];
        }
        for (int i = crossover_point; i < parent1.size(); i++)
        {
            child[i] = parent2[i];
        }
    }
    else
    {
        // If crossover doesn't occur, just return one of the parents 50-50 odds
        child = (dist(rng) < 0.5) ? parent1 : parent2;
    }

    return child;
}

// Function to perform mutation
void mutation(vector<Item>& solution)
{
    for (Item& item : solution)
    {
        // generate a random double between 0 and 1
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double mutation_chance = dist(rng);
        if (mutation_chance < MUTATION_RATE)
        {
            // Mutate the quantity (Qty) based on the available stock
            item.qty = std::min(item.qty + 1, item.stock);
        }
    }
}

// Function to generate a random initial population
vector<vector<Item>> generate_population(int population_size, const vector<Item>& items)
{
    int num_items = items.size();
    vector<vector<Item>> population(population_size);

    for (int i = 0; i < population_size; i++)
    {
        vector<Item> individual(num_items);

        for (int j = 0; j < num_items; j++)
        {
            // Randomly set the quantity for each item in the individual
            std::uniform_int_distribution<int> qty_dist(0, items[j].stock);
            individual[j].qty = qty_dist(rng);
        }

        population[i] = individual;
    }

    return population;
}


// Function to run the genetic algorithm
vector<Item> genetic_algorithm(const vector<Item>& items)
{
    auto start = std::chrono::system_clock::now();
    ofstream output_file("genetic.txt", std::ios::out | std::ios::trunc);

    vector<vector<Item>> population = generate_population(POPULATION_SIZE, items);
    for (int generation = 0; generation < NUM_GENERATIONS; generation++)
    {
        // Add fitnesses for each individual in the population
        vector<pair<double, int>> fitnesses;
        for (int i = 0; i < POPULATION_SIZE; i++)
        {
            double fitness = calculate_total_cost(population[i]);
            fitnesses.push_back({fitness, i});
        }

        // Compute the total fitness of the population
        double total_fitness = 0.0;
        for (int i = 0; i < POPULATION_SIZE; i++)
        {
            total_fitness += fitnesses[i].first;
        }

        // Compute the probabilities of selection for each individual
        vector<double> selection_probabilities(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; i++)
        {
            selection_probabilities[i] = fitnesses[i].first / total_fitness;
        }

        // Create a new population
        vector<vector<Item>> new_population(POPULATION_SIZE);

        // Generate random children by crossover
        for (int i = 0; i < POPULATION_SIZE; i++)
        {
            // Generate a random double between 0 and 1
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            // Try to clone
            if (dist(rng) < CLONING_RATE)
            {
                new_population[i] = population[fitnesses[i].second];
            }
            else
            {
                // Spin the roulette wheel to select the first parent
                double roulette_spin = dist(rng);
                double cumulative_probability = 0.0;
                int parent1_index = 0;
                for (int j = 0; j < POPULATION_SIZE; j++)
                {
                    cumulative_probability += selection_probabilities[j];
                    if (roulette_spin <= cumulative_probability)
                    {
                        parent1_index = j;
                        break;
                    }
                }

                // Spin the roulette wheel to select the second parent
                roulette_spin = dist(rng);
                cumulative_probability = 0.0;
                int parent2_index = 0;
                for (int j = 0; j < POPULATION_SIZE; j++)
                {
                    cumulative_probability += selection_probabilities[j];
                    if (roulette_spin <= cumulative_probability)
                    {
                        parent2_index = j;
                        break;
                    }
                }

                // Perform crossover and mutation to create the child
                vector<Item> child = crossover(population[parent1_index], population[parent2_index], CROSSOVER_RATE);
                mutation(child);
                new_population[i] = child;
            }
        }

        sort(fitnesses.begin(), fitnesses.end(), greater<pair<double, int>>());

        // Elitism: Always keep the best 5 solutions for each new generation
        for (int i = 0; i < 5; ++i)
        {
            new_population[i] = population[fitnesses[i].second];
        }

        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Print best individual fitnesses and time (converted to seconds)
        output_file << fitnesses[0].first << " " << (double)time / 1000000 << endl;

        population = new_population;
    }

    output_file.close();
    return population[0];
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }

    vector<Item> items = initialize_problem(argv[1]);

    // Run genetic algorithm and print solution
    auto genetic_solution = genetic_algorithm(items);

    return 0;
}
