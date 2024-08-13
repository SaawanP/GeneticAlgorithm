import math
import numpy as np


from crossover import *
from selection import *


class GA:
    def __init__(self,
                 max_generation,
                 population_size,
                 num_parents,
                 fitness_func,
                 crossover_func=single_point,
                 crossover_prob=None,
                 mutation_prob=0.05,
                 parent_selection_func="rws",
                 generational_gap=1,
                 survivor_selection_func=all_selection,
                 maximizing=True,
                 on_fitness_calc=None,
                 on_parent_selection=None,
                 on_crossover=None,
                 on_mutate=None,
                 on_survivor_selection=None,
                 on_stop=None):

        self.generation_num = 0
        self.max_generation = max_generation
        self.population_size = population_size
        self.num_parents = num_parents
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        # generational_gap is [0,1], 1 means all parents are removed, 0 means no parents are removed from next generation
        self.generational_gap = generational_gap
        self.maximizing = maximizing

        # These will hold the best solution from every generation or the current best
        self.best_solution = []
        self.best_solution_fitness = []

        self.population = []
        self.individuals_fitness = []

        self.genes_info = []

        # Store custom functions
        if callable(fitness_func) and fitness_func.__code__.co_argcount == 1:
            self.fitness_func = fitness_func
        else:
            raise ValueError(f"invalid value for fitness_func, got type: {type(fitness_func)}, value: {fitness_func}")

        if callable(survivor_selection_func) and survivor_selection_func.__code__.co_argcount == 4:
            self.survivor_selection_func = survivor_selection_func
        else:
            raise ValueError(f"invalid value for survivor_selection_func, got type: {type(survivor_selection_func)}, value: {survivor_selection_func}")

        if callable(crossover_func) and crossover_func.__code__.co_argcount == 3:
            self.crossover_func = crossover_func
        else:
            raise ValueError(f"invalid value for crossover_func, got type: {type(crossover_func)}, value: {crossover_func}")

        if callable(parent_selection_func) and parent_selection_func.__code__.co_argcount == 4:
            self.parent_select = parent_selection_func
        elif type(parent_selection_func) is str:
            if parent_selection_func == 'rws':
                self.parent_select = roulette_wheel_selection
            elif parent_selection_func == 'sus':
                self.parent_select = stochastic_universal_selection
            elif parent_selection_func == 'rbs':
                self.parent_select = rank_based_selection
            elif parent_selection_func == 'ts':
                self.parent_select = tournament_selection
            elif parent_selection_func == 'rs':
                self.parent_select = random_selection
            else:
                raise ValueError(f"parent_selection_func is not a valid string, got {parent_selection_func}")
        else:
            raise TypeError(f"parent_selection_func excepted string or function, got {type(parent_selection_func)}")

        self.on_fitness_calc = None
        if on_fitness_calc is not None:
            if callable(on_fitness_calc) and on_fitness_calc.__code__.co_argcount == 3:
                self.on_fitness_calc = on_fitness_calc

        self.on_parent_selection = None
        if on_parent_selection is not None:
            if callable(on_parent_selection) and on_parent_selection.__code__.co_argcount == 2:
                self.on_parent_selection = on_parent_selection

        self.on_crossover = None
        if on_crossover is not None:
            if callable(on_crossover) and on_crossover.__code__.co_argcount == 2:
                self.on_crossover = on_crossover

        self.on_mutate = None
        if on_mutate is not None:
            if callable(on_mutate) and on_mutate.__code__.co_argcount == 2:
                self.on_mutate = on_mutate

        self.on_survivor_selection = None
        if on_survivor_selection is not None:
            if callable(on_survivor_selection) and on_survivor_selection.__code__.co_argcount == 2:
                self.on_survivor_selection = on_survivor_selection

        self.on_stop = None
        if on_stop is not None:
            if callable(on_stop) and on_stop.__code__.co_argcount == 1:
                self.on_stop = on_stop

    def add_gene_binary(self, binary_length):
        self.genes_info.append({"type": "binary", "length": binary_length})

    # TODO currently assumes both ends of the range are given, need to add checks for None an either end of range which means inf
    def add_gene_real_discrete(self, range, precision=0.1):
        length = math.ceil(math.log2(abs(range[0] - range[1]) / precision + 1))

        self.genes_info.append({"type": "discrete", "range": range, "precision": precision, "length": length})

    def add_gene_real(self, range):
        self.genes_info.append({"type": "real", "range": range})

    def add_gene_integer(self, range):
        self.genes_info.append({"type": "integer", "range": range})

    def create_initial_population(self):
        for _ in range(self.population_size):
            genes = []
            for gene_info in self.genes_info:
                if gene_info["type"] == "binary" or gene_info["type"] == "discrete":
                    gene = np.random.randint(2 ** gene_info["length"])
                elif gene_info["type"] == "real":
                    gene = np.random.uniform(*gene_info["range"])
                elif gene_info["type"] == "integer":
                    gene = np.random.randint(*gene_info["range"])
                else:
                    raise TypeError(f"Unexpected gene_info['type'], got {gene_info['type']}")
                genes.append(gene)
            self.population.append(genes)

    def run(self):
        for i in range(self.max_generation):
            self.step()
        if self.on_stop is not None:
            self.on_stop(self)

    def step(self):
        if self.generation_num >= self.max_generation:
            print("Max number of generations reached, returning")
            return
        self.generation_num += 1

        # get fitness of population
        self.calculate_fitness()

        if self.on_fitness_calc is not None:
            self.on_fitness_calc(self, self.population, self.individuals_fitness)

        # get parents
        parents, parent_idx = self.parent_select(self.population, self.individuals_fitness, self.num_parents, self.maximizing)

        if self.on_parent_selection is not None:
            self.on_parent_selection(self, parents)

        # crossover for children
        children = self.crossover_func(parents, self.crossover_prob, self.genes_info)

        if self.on_crossover is not None:
            self.on_crossover(self, children)

        # mutate children
        children = self.mutate(children)

        if self.on_mutate is not None:
            self.on_mutate(self, children)

        # select survivors
        fitness = np.array([self.individuals_fitness[i] for i in parent_idx])
        idx_sort = fitness.argsort()
        if self.maximizing:
            parents = parents[idx_sort[::-1]]
        else:
            parents = parents[idx_sort]
        self.population = self.survivor_selection_func(parents, children, self.generational_gap)

    def calculate_fitness(self):
        self.individuals_fitness = []
        for individual in self.population:
            fitness = self.fitness_func(individual)
            self.individuals_fitness.append(fitness)

    def mutate(self, children):
        for child in children:
            for gene_num, gene_info in enumerate(self.genes_info):
                new_gene = child[gene_num]
                if gene_info["type"] == "binary" or gene_info["type"] == "discrete":
                    for bit in range(gene_info["length"]):
                        mutate = np.random.uniform()
                        if mutate <= self.mutation_prob:
                            new_gene = new_gene ^ 2 ** bit  # flip bit
                    if gene_info["type"] == "discrete":  # check if new mutation in within range, otherwise clamp it
                        if new_gene * gene_info["precision"] > abs(gene_info["range"][1] - gene_info["range"][0]):
                            new_gene = int(gene_info["range"][1] - gene_info["range"][0])
                # TODO add a variation for random noise
                elif gene_info["type"] == "real":
                    mutate = np.random.uniform()
                    if mutate <= self.mutation_prob:
                        new_gene = np.random.uniform(*gene_info["range"])  # new random number within range
                # TODO add a variation for smoother integer noise
                elif gene_info["type"] == "integer":
                    mutate = np.random.uniform()
                    if mutate <= self.mutation_prob:
                        new_gene = np.random.randint(*gene_info["range"])  # new random number within range
                child[gene_num] = new_gene
        return children
