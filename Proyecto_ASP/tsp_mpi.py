####################################################################################################
# Proyecto de Arquitectura de Sistemas Paralelos                                                   #
# Paralelización de un Algoritmo Genético aplicado al Problema del Viajante                        #
#                                                                                                  #
# tsp_mpi.py: GA para TSP paralelizado con MPI                                                     #
# Pareja 3, Grupo 2461:                                                                            #
#       Michael Alexander Fajardo                                                                  #
#       Gloria del Valle Cano                                                                      #
####################################################################################################
from tsp import *
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from mpi4py.futures import MPICommExecutor
from functools import partial
from deap import algorithms, base, creator, tools

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


if __name__ == "__main__":

	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	random.seed(4)
	N_ISLES = size
	FREQ = 5
	
	if rank == 0:
		pob = int(500/N_ISLES)	
		islands = [toolbox.population(n=pob) for i in range(N_ISLES)]
	else:
		islands = None


	toolbox.unregister("indices")
	toolbox.unregister("individual")
	toolbox.unregister("population")

	if rank == 0 :
		start_time = time.time()
	for i in range (0, 400, FREQ):
		islands = comm.scatter(islands,root = 0)
		resultsT =  algorithms.eaSimple(islands,toolbox=toolbox, cxpb=0.8, mutpb=0.2, ngen=5, verbose=False)
		results = comm.gather(resultsT,root=0)
		if rank== 0:
			islands = [pop for pop, logbook in results]
			tools.migRing(islands, 15, tools.selBest)

	if rank == 0:
		print("--- %s seconds ---" % (time.time() - start_time))
