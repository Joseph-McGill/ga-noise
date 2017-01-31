import genetic_algorithm
import matplotlib.pyplot as plt

## Joseph McGill
## Fall 2016
## This is the main program. It runs genetic algorithms using rank-based
## and tournament selection on the TSP instances in the tsp_instances variables

## Note that this program takes a considerable amount of time to run

# Main Function
if __name__ == "__main__":
    # TSP instances available for use
    tsp_instances = ['burma14', 'ulysses16', 'ulysses22', 'bays29',
                    'dantzig42', 'att48', 'eil51' , 'eil76']

    # optimal solutions for each instance
    opt_distances = {'burma14': 3323, 'bays29': 2020, 'dantzig42': 699,
                    'eil51': 426, 'ulysses16': 6859, 'ulysses22': 7013,
                    'att48': 10628, 'eil76': 538}

    # TSP instances used in experiments
    tsp_instances = ['burma14', 'ulysses16', 'ulysses22']

    # tournament sizes used in experiments
    tournament_sizes = [2, 4, 6, 8, 10]

    # selection pressures used in experiments
    selection_pressures = [1.1, 1.3, 1.5, 1.7 ,2.0]


    tsp_instances = ['ulysses22']
    results = {}

    # run the experiment
    for instance in tsp_instances:

        tournament_lowest_distances = []
        ranked_lowest_distances = []

        # create the genetic algorithm for the current instance
        ga = genetic_algorithm.GA(instance)

        if False:
            # run the tournament selection
            for size in tournament_sizes:
                tournament_lowest_distances.append(ga.run_tournament(
                                                tourn_size = size))

        if True:
            # run the ranked based selection
            for pressure in selection_pressures:
                ranked_lowest_distances.append(ga.run_ranked(
                                                selection_pressure = pressure))

            results[instance] = [tournament_lowest_distances,
                                ranked_lowest_distances]

# output the results
for index, instance in enumerate(tsp_instances):

    # output the tournament selection results
    if False:
        tournament_results = []
        for i in range(4):
            tournament_results.append([x[i] for x in results[instance][0]])

        plt.plot([2, 10], [opt_distances[instance], opt_distances[instance]],
                'r--', label='Optimal solution')

        plt.plot(tournament_sizes, tournament_results[0],'b-o',
                label='Noiseless')
        plt.plot(tournament_sizes, tournament_results[1],'g-^',
                label='var(population) noise')

        plt.plot(tournament_sizes, tournament_results[2],'m-*',
                label='2 * var(population) noise')

        plt.plot(tournament_sizes, tournament_results[3],'k-s',
                label='4 * var(population) noise')

        plt.legend(loc="upper right")
        title = instance + ' Tournament results'
        plt.title(title)
        plt.xlabel('Tournament Sizes')
        plt.ylabel('Distance')
        plt.ylim([opt_distances[instance] - 1000,
                opt_distances[instance] + 1000])
        plt.show()

    # output the rank-based selection results
    if True:

        ranked_results = []
        for i in range(4):
            ranked_results.append([x[i] for x in results[instance][1]])

        plt.plot([1.1, 2.0], [opt_distances[instance], opt_distances[instance]],
                 'r--', lw = 2, label='Optimal solution')

        plt.plot(selection_pressures, ranked_results[0],'b-o',
                label='Noiseless')
        plt.plot(selection_pressures, ranked_results[1],'g-^',
                label='var(population) noise')

        plt.plot(selection_pressures, ranked_results[2],'m-*',
                label='2 * var(population) noise')

        plt.plot(selection_pressures, ranked_results[3],'k-s',
                label='4 * var(population) noise')

        plt.legend(loc="upper right")
        title = instance + ' Ranked results'
        plt.title(title)
        plt.xlabel('Selection Pressures')
        plt.ylabel('Distance')
        plt.ylim([opt_distances[instance] - 1000,
                opt_distances[instance] + 5000])

        plt.show()
