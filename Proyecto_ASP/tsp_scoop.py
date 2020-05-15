####################################################################################################
# Proyecto de Arquitectura de Sistemas Paralelos                                                   #
# Paralelización de un Algoritmo Genético aplicado al Problema del Viajante                        #
#                                                                                                  #
# tsp_scoop.py: GA para TSP paralelizado con SCOOP                                                 #
# Pareja 3, Grupo 2461:                                                                            #
#       Michael Alexander Fajardo                                                                  #
#       Gloria del Valle Cano                                                                      #
####################################################################################################
from tsp import *
from functools import partial
from deap import algorithms, base, creator, tools
from scoop import futures
import sys
num_cities = 30
cities = generate_cities(num_cities)

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("indices", numpy.random.permutation, len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

def create_tour(individual):
    return [list(cities)[e] for e in individual]

def evaluation(individual):
    '''Evaluates an individual by converting it into
    a list of cities and passing that list to total_distance'''
    return (total_distance(create_tour(individual)),)

toolbox.register("evaluate", evaluation)

toolbox.register("select", tools.selTournament, tournsize=3)

#toolbox.register("map", futures.map)

def main(number):

	random.seed(4)
	N_ISLES = number
	FREQ = 5
	pob = int(500/number)
	islands = [toolbox.population(n=pob) for i in range(N_ISLES)]

	toolbox.unregister("indices")
	toolbox.unregister("individual")
	toolbox.unregister("population")

	toolbox.register("alg_scoop", algorithms.eaSimple, toolbox=toolbox, cxpb=0.8, mutpb=0.2, ngen=5, verbose=False)

	start_time = time.time()
	for i in range (0, 400, FREQ):
		results = futures.map(toolbox.alg_scoop, islands)
		islands = [pop for pop, logbook in results]
		tools.migRing(islands, 15, tools.selBest)

	print("--- %s seconds ---" % (time.time() - start_time))
	return "finished"
    #return islands

if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		sys.exit()

	print(main(int(sys.argv[1])))
