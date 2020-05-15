####################################################################################################
# Proyecto de Arquitectura de Sistemas Paralelos                                                   #
# Paralelización de un Algoritmo Genético aplicado al Problema del Viajante                        #
#                                                                                                  #
# tsp.py: Funciones TSP básicas                                                                    #
# Pareja 3, Grupo 2461:                                                                            #
#       Michael Alexander Fajardo                                                                  #
#       Gloria del Valle Cano                                                                      #
####################################################################################################
import random, operator, time, itertools, math
import numpy
import array

City = complex # Constructor for new cities, e.g. City(300, 400)

def total_distance(tour):
    "The total distance between each pair of consecutive cities in the tour."
    return sum(distance(tour[i], tour[i-1])
               for i in range(len(tour)))

def exact_TSP(cities):
    "Generate all possible tours of the cities and choose the shortest one."
    return shortest(alltours(cities))

def shortest(tours):
    "Return the tour with the minimum total distance."
    return min(tours, key=total_distance)

def distance(A, B):
    "The Euclidean distance between two cities."
    return abs(A - B)

def generate_cities(n):
    "Make a set of n cities, each with random coordinates."
    return set(City(random.randrange(10, 890),
                    random.randrange(10, 590))
               for c in range(n))

def all_non_redundant_tours(cities):
    "Return a list of tours, each a permutation of cities, but each one starting with the same city."
    start = first(cities)
    return [[start] + list(tour)
            for tour in itertools.permutations(cities - {start})]

def first(collection):
    "Start iterating over collection, and return the first element."
    for x in collection: return x

def exact_non_redundant_TSP(cities):
    "Generate all possible tours of the cities and choose the shortest one."
    return shortest(all_non_redundant_tours(cities))

def greedy_TSP(cities):
    "At each step, visit the nearest neighbor that is still unvisited."
    start = first(cities)
    tour = [start]
    unvisited = cities - {start}
    while unvisited:
        C = nearest_neighbor(tour[-1], unvisited)
        tour.append(C)
        unvisited.remove(C)
    return tour

def nearest_neighbor(A, cities):
    "Find the city in cities that is nearest to city A."
    return min(cities, key=lambda x: distance(x, A))
