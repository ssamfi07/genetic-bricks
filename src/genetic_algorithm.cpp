#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <unordered_map>

using namespace std;

// Data structure for an item
struct Item
{
    string itemNo;
    string colorID;
    int qty;
    double weight;
    double volume;
    std::vector<bool> status;
    std::vector<int> stock;
    std::vector<double> price;
    // std::vector<string> store;
    std::vector<string> country;
    std::vector<double> minValor;
    std::vector<double> free;
    std::vector<double> racio;
};

// Constants for genetic algorithm
const int POPULATION_SIZE = 100;
const int NUM_GENERATIONS = 100;
const double CROSSOVER_RATE = 0.75;
const double MUTATION_RATE = 0.01;
const double CLONING_RATE = 0.05;

// Random number generator
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

void displayItem(const Item& item)
{
    std::cout << "Item No: " << item.itemNo << std::endl;
    std::cout << "Color ID: " << item.colorID << std::endl;
    std::cout << "Quantity: " << item.qty << std::endl;
    std::cout << "Weight: " << item.weight << std::endl;
    std::cout << "Volume: " << item.volume << std::endl;

    std::cout << "Status: [ ";
    for (const auto& s : item.status)
    {
        std::cout << s << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Stock: [ ";
    for (const auto& s : item.stock)
    {
        std::cout << s << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Price: [ ";
    for (const auto& p : item.price)
    {
        std::cout << p << ", ";
    }
    std::cout << "]" << std::endl;

    // std::cout << "Store: [ ";
    // for (const auto& s : item.store)
    // {
    //     std::cout << "'" << s << "' ";
    // }
    // std::cout << "]" << std::endl;

    std::cout << "Country: [ ";
    for (const auto& c : item.country)
    {
        std::cout << c << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Min Valor: [ ";
    for (const auto& m : item.minValor)
    {
        std::cout << m << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Free: [ ";
    for (const auto& f : item.free)
    {
        std::cout << f << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Racio: [ ";
    for (const auto& r : item.racio)
    {
        std::cout << r << ", ";
    }
    std::cout << "]" << std::endl;
}

vector<Item> initialize_problem(string filename)
{
    std::vector<Item> items;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open input file." << std::endl;
        return items;
    }

    std::string line;

    // Skip the header line
    getline(file, line);

    // Read file line by line
    while (getline(file, line))
    {
        // Create a stringstream from the line
        std::istringstream ss(line);
        std::string token;

        // Initialize Item object
        Item item;

        // Read and parse each token
        getline(ss, item.itemNo, ';');
        // std::cout << item.itemNo << std::endl;
        getline(ss, item.colorID, ';');
        ss >> item.qty;
        ss.ignore(); // Ignore the delimiter
        ss >> item.weight;
        ss.ignore(); // Ignore the delimiter
        ss >> item.volume;
        ss.ignore(); // Ignore the delimiter

        // Read and assign values to the status vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream status_ss(token);

        // Loop to read integers from status_ss and add them to item.status
        while (status_ss >> token)
        {
            try
            {
                item.status.push_back(stoi(token));
            }
            catch (const std::invalid_argument& e)
            {
                // Print the faulty token before exiting
                std::cerr << "Error converting token to integer: " << token << std::endl;
                // You can handle the error or exit the program here
                exit(EXIT_FAILURE);
            }

            // Check for the comma, if present, ignore it
            if (status_ss.peek() == ',')
                status_ss.ignore();
        }

        ss.ignore(); // Ignore the delimiter
        // read the Stock vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream stock_ss(token);

        // Loop to read integers from stock_ss and add them to item.stock
        while (stock_ss >> token)
        {
            try
            {
                item.stock.push_back(stoi(token));
            }
            catch (const std::invalid_argument& e)
            {
                // Print the faulty token before exiting
                std::cerr << "Error converting token to integer: " << token << std::endl;
                // You can handle the error or exit the program here
                exit(EXIT_FAILURE);
            }

            // Check for the comma, if present, ignore it
            if (stock_ss.peek() == ',')
                stock_ss.ignore();
        }

        ss.ignore(); // Ignore the delimiter
        // read the Prices vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream price_ss(token);

        // Loop to read integers from price_ss and add them to item.prices
       while (price_ss >> token)
        {
            try
            {
                item.price.push_back(stod(token));
            }
            catch (const std::invalid_argument& e)
            {
                // Print the faulty token before exiting
                std::cerr << "Error converting token to integer: " << token << std::endl;
                // You can handle the error or exit the program here
                exit(EXIT_FAILURE);
            }

            // Check for the comma, if present, ignore it
            if (price_ss.peek() == ',')
                price_ss.ignore();
        }

        ss.ignore(); // Ignore the delimiter
        // Read and assign values to the countries vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream countries_ss(token);

        while (getline(countries_ss, token, ','))
        {
            // Remove leading and trailing whitespaces from the token
            token.erase(0, token.find_first_not_of(" \t\n\r\f\v'"));
            token.erase(token.find_last_not_of(" \t\n\r\f\v'") + 1);

            item.country.push_back(token);
        }

        ss.ignore(); // Ignore the delimiter
        // read the minValor vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream minval_ss(token);

        // Loop to read integers from minval_ss and add them to item.minvalor
        while (minval_ss >> token)
        {
            item.minValor.push_back(stod(token));
            // Check for the comma, if present, ignore it
            if (minval_ss.peek() == ',')
                minval_ss.ignore();
        }

        ss.ignore(); // Ignore the delimiter
        // read the free vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream free_ss(token);

        // Loop to read integers from free_ss and add them to item.free
        while (free_ss >> token)
        {
            item.free.push_back(stod(token));
            // Check for the comma, if present, ignore it
            if (free_ss.peek() == ',')
                free_ss.ignore();
        }

        ss.ignore(); // Ignore the delimiter
        // read the racio vector
        getline(ss, token, '[');
        getline(ss, token, ']');
        istringstream racio_ss(token);

        // Loop to read integers from racio_ss and add them to item.racio
        while (racio_ss >> token)
        {
            item.racio.push_back(stod(token));
            // Check for the comma, if present, ignore it
            if (racio_ss.peek() == ',')
                racio_ss.ignore();
        }

        ss.ignore(); // Ignore the delimiter

        items.push_back(item);
        // displayItem(item);
    }

    file.close();
    return items;
}


// Function to randomly choose qty elements from each vector
template <typename T>
void randomlyChooseElements(std::vector<T>& vec, size_t qty)
{
    // !! TODO
    /* Some sellers require a minimum purchase value of parts per order (𝑚𝑖𝑛𝑗 ) 
    and a minimum average value per SKU (𝑟𝑎𝑐𝑖𝑜𝑗 ).*/

    // Ensure that qty is not greater than the vector size
    qty = std::min(qty, vec.size());

    // Shuffle indices
    std::vector<size_t> indices(vec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    // Sort the first qty indices to get random indices
    std::sort(indices.begin(), indices.begin() + qty);

    // Create a temporary vector to store the selected elements
    std::vector<T> selectedElements(qty);

    // Copy the selected elements from the original vector
    for (size_t i = 0; i < qty; ++i) {
        selectedElements[i] = vec[indices[i]];
    }

    // Assign the selected elements back to the original vector
    vec = selectedElements;
}

// Function to randomly choose elements for each item
void randomlyChooseElementsForItem(Item& item, size_t qty) {
    randomlyChooseElements(item.status, qty);
    randomlyChooseElements(item.stock, qty);
    randomlyChooseElements(item.price, qty);
    randomlyChooseElements(item.country, qty);
    randomlyChooseElements(item.minValor, qty);
    randomlyChooseElements(item.free, qty);
    randomlyChooseElements(item.racio, qty);
}

// Helper function to parse the lower and upper bounds of a range
std::pair<int, int> parseRange(const std::string& range)
{
    std::pair<int, int> result;
    std::istringstream ss(range);

    // Read the lower bound
    ss >> result.first;

    // Ignore non-digit characters until a digit is encountered
    ss.ignore(std::numeric_limits<std::streamsize>::max(), '-');

    // Read the upper bound
    ss >> result.second;

    return result;
}

double calculateShippingCost(const std::string& country, double weight,
                                const std::vector<std::string>& ranges,
                                const std::vector<std::string>& countries,
                                const std::vector<std::vector<double>>& shippingCosts)
{
    // Find the index of the country in the 'countries' vector
    auto countryIndex = std::find(countries.begin(), countries.end(), country);

    

    // Check if the country is found
    if (countryIndex != countries.end())
    {
        // std::cout << "Index found" << std::endl;
        // Calculate the total cost based on the weight range
        auto colIndex = std::distance(countries.begin(), countryIndex);
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            // get the ranges interval
            auto rangesPair = parseRange(ranges[i]);
            // decide the weight range
            if (weight <= rangesPair.second && weight >= rangesPair.first)
            {
                // std::cout << "Lower Bound: " << rangesPair.first << std::endl;
                // std::cout << "Upper Bound: " << rangesPair.second << std::endl;
                // std::cout << colIndex << " " << i << std::endl;
                // std::cout << shippingCosts[i][colIndex] << std::endl;
                // return the shipping cost
                return shippingCosts[i][colIndex];
            }
        }

        // If the weight exceeds the highest range, use the cost from the last column
        return shippingCosts[ranges.size() - 1].back();
    } 
    else
    {
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            // get the ranges interval
            auto rangesPair = parseRange(ranges[i]);
            // decide the weight range
            if (weight <= rangesPair.second && weight >= rangesPair.first)
            {
                // std::cout << "Lower Bound: " << rangesPair.first << std::endl;
                // std::cout << "Upper Bound: " << rangesPair.second << std::endl;
                // std::cout << colIndex << " " << i << std::endl;
                // std::cout << shippingCosts[i][colIndex] << std::endl;
                // return the shipping cost for ALl Other
                return shippingCosts[i].back();
            }
        }
        // If the weight exceeds the highest range, use the cost from the last column
        return shippingCosts[ranges.size() - 1].back();
    }
}

void readShippingCosts(const std::string& filename,
                        std::vector<std::string>& ranges,
                        std::vector<std::string>& countries,
                        std::vector<std::vector<double>>& shippingCosts)
{
    // Open the CSV file
    std::ifstream file(filename);

    // Check if the file is open
    if (file.is_open())
    {
        std::string line;

        // Read the header line
        std::getline(file, line);

        // Use a stringstream to parse the header line
        std::istringstream headerStream(line);

        // Skip the first column
        std::string range;
        std::getline(headerStream, range, ';');

        // Read and store the countries
        while (std::getline(headerStream, range, ';')) {
            countries.push_back(range);
        }

        // Read the remaining lines
        while (std::getline(file, line)) {
            // Use a stringstream to parse each line
            std::istringstream ss(line);

            // Read the ranges
            std::getline(ss, range, ';');
            ranges.push_back(range);

            // Read and store values for each country
            std::vector<double> countryCosts;
            double cost;
            while (ss >> cost) {
                countryCosts.push_back(cost);

                // Skip the semicolon
                if (ss.peek() == ';')
                    ss.ignore();
            }

            // Store the country costs
            shippingCosts.push_back(countryCosts);
        }

        // Close the file
        file.close();
    } else {
        std::cerr << "Error: Unable to open the file." << std::endl;
    }
}

// Function to calculate the fitness of an individual solution (to be minimized)
double calculate_total_cost(const vector<Item>& items,
                            const std::vector<std::string>& ranges,
                            const std::vector<std::string>& countries,
                            const std::vector<std::vector<double>>& shippingCosts)
{
    double total_cost = 0.0;

    std::unordered_map<std::string, double> country_weights; // Track total weights per country

    for (const Item& item : items)
    {
        // simply add the prices of the variants to the total cost of a solution
        total_cost += std::accumulate(item.price.begin(), item.price.end(), 0);
        // add the country weights to the dictionary of country weights
        for (int i = 0; i < item.qty; ++i)
        {
            // take the country name of one variant
            std::string country = item.country[i];
            // add it's weight to the dictionary entry
            country_weights[country] += item.weight;
        }
        // magic happens here

        // the constraints, prices, shipping, etc etc
    }

    // add the shipping prices for each country
    // Print total weights per country (optional)
    for (const auto& entry : country_weights)
    {
        // std::cout << "Total weight for " << entry.first << ": " << entry.second << " grams\n";
        auto cost = calculateShippingCost(entry.first, entry.second, ranges, countries, shippingCosts);
        // std::cout << "Shipping cost to " << entry.first << " for weight " << entry.second << ": " << cost << std::endl;
        total_cost += cost;
    }
    return total_cost;
}

// Function to perform crossover between 2 individual solutions
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

// Function to perform mutation on the child
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
            for (auto& item : solution) {
                randomlyChooseElementsForItem(item, item.qty);
                // displayItem(item);
            }
        }
    }
}

// Function to generate a random initial population of individual (possible) solutions
vector<vector<Item>> generate_population(int population_size, vector<Item> items)
{
    // initialize the population
    vector<vector<Item>> population(population_size);

    for (int i = 0; i < population_size; i++)
    {
        // create random individuals
        vector<Item> individual = items;
        // Apply the function to each item
        for (auto& item : individual) {
            randomlyChooseElementsForItem(item, item.qty);
            // displayItem(item);
        }

        population[i] = individual;
    }
    return population;
}


// Function to run the genetic algorithm
vector<Item> genetic_algorithm(const vector<Item>& items)
{
    auto start = std::chrono::system_clock::now();
    // ofstream output_file("genetic.txt", std::ios::out | std::ios::trunc);

    vector<vector<Item>> population = generate_population(POPULATION_SIZE, items);
    for (int generation = 0; generation < NUM_GENERATIONS; generation++)
    {
        // Add fitnesses for each individual in the population
        vector<pair<double, int>> fitnesses;
        for (int i = 0; i < POPULATION_SIZE; i++)
        {
            // double fitness = calculate_total_cost(population[i]);
            // fitnesses.push_back({fitness, i});
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
                // mutation(child); -- for later
                new_population[i] = child;
            }
        }

        sort(fitnesses.begin(), fitnesses.end(), greater<pair<double, int>>());

        // Elitism: Always keep the best 5 solutions for each new generation
        for (int i = POPULATION_SIZE-1; i >= 0; i--)
        {
            new_population[i] = population[fitnesses[i].second];
        }

        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Print best individual fitnesses and time (converted to seconds)
        // output_file << fitnesses[0].first << " " << (double)time / 1000000 << endl;

        population = new_population;
    }

    // output_file.close();
    return population[0];
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    std::vector<std::string> ranges;
    std::vector<std::string> countries;
    std::vector<std::vector<double>> shippingCosts;

    std::vector<Item> items = initialize_problem(argv[1]);


    readShippingCosts("../input/ShippingRates.csv", ranges, countries, shippingCosts);

    // Print the vectors for verification
    // std::cout << "Ranges: ";
    // for (const auto& range : ranges) {
    //     std::cout << range << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Countries: ";
    // for (const auto& country : countries) {
    //     std::cout << country << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Shipping Costs:" << std::endl;
    // for (const auto& row : shippingCosts) {
    //     for (const auto& cost : row) {
    //         std::cout << cost << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // create random individuals
    vector<Item> individual = items;
    // Apply the function to each item
    for (auto& item : individual) {
        randomlyChooseElementsForItem(item, item.qty);
        // displayItem(item);
    }

    std::cout << calculate_total_cost(individual, ranges, countries, shippingCosts) << std::endl;


    // auto start = std::chrono::system_clock::now();
    // auto genetic_solution = genetic_algorithm(items);
    // auto end = std::chrono::system_clock::now();
    // auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "Time taken: " << time << " microseconds" << std::endl;
    

    return 0;
}
