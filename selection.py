"""
Parent selection functions
"""


def roulette_wheel_selection(population, fitness, num_parents, maximizing):
    pass


def stochastic_universal_selection(population, fitness, num_parents, maximizing):
    pass


def rank_based_selection(population, fitness, num_parents, maximizing):
    pass


def tournament_selection(population, fitness, num_parents, maximizing):
    pass


def random_selection(population, fitness, num_parents, maximizing):
    pass


"""
Survivor selection functions
"""


def all_selection(parents, children, generation_gap):
    return parents + children


def elitism_selection(parents, children, generation_gap):
    return parents[:int(len(parents)*(1-generation_gap))] + children
