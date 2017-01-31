import json
import copy
import array
import random
import numpy as np
from deap import algorithms, base, creator, tools
from operator import attrgetter

## Joseph McGill
## Fall 2016
## This is a simple genetic algorithm implementation. It uses linear order
## crossover and shuffle mutation along with rank-based and tournament
## selection. The DEAP library was used.

# Simple Genetic Algorithm
class GA:

    # dict for parameters given an instance
    tsp_instances = {'burma14': [100, 140], 'bays29': [800, 290],
                    'dantzig42': [900, 420], 'eil51': [1000, 510],
                     'ulysses16': [150, 160], 'ulysses22': [200, 220],
                     'att48': [400, 480], 'eil76': [700, 760]}

    # create the fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # create the individual container
    creator.create("Individual", array.array, typecode='i',
                    fitness=creator.FitnessMin)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    # Constructor
    def __init__(self, tsp_instance = 'burma14'):

        # set the class instance variables
        self.tsp_instance = tsp_instance
        self.NUM_GENERATIONS = GA.tsp_instances[self.tsp_instance][0]
        self.POP_SIZE = GA.tsp_instances[self.tsp_instance][1]

        # open the TSP instance and load it into a dict
        with open('../data/' + self.tsp_instance + '.json') as tsp_data:
            self.tsp = json.load(tsp_data)

        # get the distance matrix and tour size from the json object
        self.distance_matrix = self.tsp["DistanceMatrix"]
        self.tour_size = self.tsp["TourSize"]

        # create the random individual generator
        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", random.sample,
                             range(self.tour_size), self.tour_size)

        # create the individual and population initializers
        self.toolbox.register("individual", tools.initIterate,
                             creator.Individual, self.toolbox.indices)

        self.toolbox.register("population", tools.initRepeat,
                             list, self.toolbox.individual)

        # define the default genetic operations for the GA
        self.toolbox.register("select", tools.selTournament, tournsize = 2)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 0.05)
        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("evaluate", self.evalTSP)


    # Function to run the GA once using Tournament selection
    def run_tournament(self, tourn_size = 2, cross_prob = 0.6, mut_prob = 0.1):

        noise_levels = [0, 1, 2, 4]

        # generate the random populations
        populations = []
        for i in range(10):
            populations.append(self.toolbox.population(n = self.POP_SIZE))

        noise_populations = []
        for i in range(len(noise_levels)):
            noise_populations.append(copy.deepcopy(populations))

        lowest_distances = []

        # for each noise level run 10 trials and take lowest distance as result
        for index, noise_level in enumerate(noise_levels):

            # set the selection to the appropriate noise level
            self.toolbox.register("select", GA.selTournament,
                         tournsize = tourn_size, noise_factor = noise_level)

            lowest_distance = np.inf
            for trial, pop in enumerate(noise_populations[index]):
                algorithms.eaSimple(pop, self.toolbox, cross_prob, mut_prob,
                        self.NUM_GENERATIONS, stats = GA.stats, verbose = False)

                new_min = GA.stats.compile(pop)['min']
                if new_min < lowest_distance:
                    lowest_distance = new_min

            lowest_distances.append(lowest_distance)
            print("GA for %s, tournament size: %d, noise level %d"
                    % (self.tsp_instance, tourn_size, noise_level))

            print("Optimal Distance: %d" % self.tsp["OptDistance"])
            print("Min distance found: %d\n" % lowest_distance)

        # return the statistics of the final population
        return lowest_distances


    # Function to run the GA once using Rank-Based Roulette Wheel selection
    def run_ranked(self, selection_pressure = 1.1,
                cross_prob = 0.6, mut_prob = 0.1):

        noise_levels = [0, 1, 2, 4]

        # generate the random populations
        populations = []
        for i in range(10):
            populations.append(self.toolbox.population(n = self.POP_SIZE))

        noise_populations = []
        for i in range(len(noise_levels)):
            noise_populations.append(copy.deepcopy(populations))

        lowest_distances = []

        # for each noise level run 10 trials and take lowest distance as result
        for index, noise_level in enumerate(noise_levels):

            # set the selection to the appropriate noise level
            self.toolbox.register("select", GA.selRankBased,
               select_pressure = selection_pressure, noise_factor = noise_level)

            lowest_distance = np.inf
            for trial, pop in enumerate(noise_populations[index]):
                algorithms.eaSimple(pop, self.toolbox, cross_prob,
                mut_prob, self.NUM_GENERATIONS, stats = GA.stats,
                verbose = False)

                new_min = GA.stats.compile(pop)['min']
                if new_min < lowest_distance:
                    lowest_distance = new_min

            lowest_distances.append(lowest_distance)
            print("GA for %s, selection pressure: %.1f, noise level %d"
                    % (self.tsp_instance, selection_pressure, noise_level))

            print("Optimal Distance: %d" % self.tsp["OptDistance"])
            print("Min distance found: %d\n" % lowest_distance)

        # return the statistics of the final population
        return lowest_distances

    # Function to evaluate an individual
    def evalTSP(self, individual):
        distance = self.distance_matrix[individual[-1]][individual[0]]
        for gene1, gene2 in zip(individual[0:-1], individual[1:]):
            distance += self.distance_matrix[gene1][gene2]
        return distance,


    # overload of DEAP's selTournament function to accomodate noise
    def selTournament(individuals, k, tournsize, noise_factor = 0):

        # add noise if necessary
        if noise_factor > 0:

            # add some noise to the fitness values
            pop_std = np.std([ind.fitness.values[0] for ind in individuals])
            if (pop_std == 0):
                pop_std = 1
            noise = np.random.normal(0,  np.sqrt(noise_factor) * pop_std,
                                    len(individuals))
            noisy_individuals = []

            for index, ind in enumerate(individuals):
                noisy_individuals.append((ind,
                                ind.fitness.values[0] + noise[index]))

            chosen = []
            for i in range(k):
                aspirants = tools.selRandom(individuals, tournsize)
                indices = [individuals.index(j) for j in aspirants]
                best = min([noisy_individuals[j] for j in indices],
                            key = lambda x:x[1])

                chosen.append(best[0])
            return chosen

        else:

            # normal tournament selection
            chosen = []

            for i in range(k):

                aspirants = tools.selRandom(individuals, tournsize)
                chosen.append(max(aspirants, key=attrgetter("fitness")))
            return chosen

    # Rank based roulette wheel selection
    def selRankBased(individuals, k, select_pressure, noise_factor = 0):

        if noise_factor > 0:

            # add some noise to the fitness values
            pop_std = np.std([ind.fitness.values[0] for ind in individuals])
            if (pop_std == 0):
                pop_std = 1
            noise = np.random.normal(0,  np.sqrt(noise_factor) * pop_std,
                                     len(individuals))
            noisy_individuals = []

            for index, ind in enumerate(individuals):
                noisy_individuals.append((ind,
                                    ind.fitness.values[0] + noise[index]))

            # sort the noisy individuals
            sorted_noisy = sorted(noisy_individuals, key = lambda x:x[1])
            s_inds = [ind[0] for ind in sorted_noisy]

        else:
            s_inds = sorted(individuals,
                        key = attrgetter('fitness'), reverse = True)

        # rank the individuals
        ranked_inds = []
        for index, ind in enumerate(s_inds):
            rank = (2 - select_pressure + (2*(select_pressure - 1)
                    *((len(s_inds) - index - 1)/(len(s_inds) - 1))))

            ranked_inds.append((ind, rank))

        # sum the ranks
        sum_ranks = sum(x[1] for x in ranked_inds)

        # select k individuals
        chosen = []
        for i in range(k):

            # spin the wheel
            u = random.random() * sum_ranks

            # select the individual where the spin lands
            sum_ = 0
            for ind in ranked_inds:
                sum_ += ind[1]
                if sum_ > u:
                    chosen.append(ind[0])
                    break

        # return the chosen individuals
        return chosen
