import math
import numpy

from crossover import *
from parent_select import *


class GA:
    def __init__(self,
                 max_generation,
                 population_size,
                 num_parents,
                 fitness_func,
                 crossover_func=single_point,
                 crossover_prob=1,
                 mutation_prob=0.05,
                 generational_gap=1,
                 parent_selection_type="rws",
                 on_fitness_calc=None,
                 on_parent_selection=None,
                 on_crossover=None,
                 on_mutate=None,
                 on_survivor_selection=None):

        self.generation_num = 0
        self.max_generation = max_generation
        self.population_size = population_size
        self.num_parents = num_parents
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generational_gap = generational_gap

        # These will hold the best solution from every generation or the current best
        self.best_solution = []
        self.best_solution_fitness = []

        self.population = []
        self.individuals_fitness = []

        self.genes_info = []

        # Store custom functions
        self.on_fitness_calc = None
        if on_fitness_calc is not None:
            if callable(on_fitness_calc) and on_fitness_calc.__code__.co_argcount == 1:
                self.on_fitness_calc = on_fitness_calc

        self.on_parent_selection = None
        if on_parent_selection is not None:
            if callable(on_parent_selection) and on_parent_selection.__code__.co_argcount == 1:
                self.on_parent_selection = on_parent_selection

        self.on_crossover = None
        if on_crossover is not None:
            if callable(on_crossover) and on_crossover.__code__.co_argcount == 1:
                self.on_crossover = on_crossover

        self.on_mutate = None
        if on_mutate is not None:
            if callable(on_mutate) and on_mutate.__code__.co_argcount == 1:
                self.on_mutate = on_mutate

        self.on_survivor_selection = None
        if on_survivor_selection is not None:
            if callable(on_survivor_selection) and on_survivor_selection.__code__.co_argcount == 1:
                self.on_survivor_selection = on_survivor_selection

        if callable(parent_selection_type) and parent_selection_type.__code__.co_argcount == 3:
            self.parent_select = parent_selection_type
        elif type(parent_selection_type) is str:
            if parent_selection_type == 'rws':
                self.parent_select = roulette_wheel_selection
        else:
            raise TypeError(f"parent_selection_type excepted string, got {type(parent_selection_type)}")

    def add_gene_binary(self, binary_length):
        self.genes_info.append({"type": "binary", "length": binary_length})

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
                    gene = numpy.random.randint(2 ** gene_info["length"])
                elif gene_info["type"] == "real":
                    gene = numpy.random.uniform(*gene_info["range"])
                elif gene_info["type"] == "integer":
                    gene = numpy.random.randint(*gene_info["range"])
                else:
                    raise TypeError(f"Unexpected gene_info['type'], got {gene_info['type']}")
                genes.append(gene)
            self.population.append(genes)

    def run(self):
        for i in range(self.max_generation):
            self.step()

    def step(self):
        if self.generation_num >= self.max_generation:
            print("Max number of generations reached, returning")
            return
        self.generation_num += 1

        # get fitness of population
        self.individuals_fitness = []
        for individual in self.population:
            fitness = self.fitness_func(individual)
            self.individuals_fitness.append(fitness)

        if self.on_fitness_calc is not None:
            self.on_fitness_calc(self, self.population, self.individuals_fitness)

        # get parents
        parents = self.parent_select(self.population, self.individuals_fitness, self.num_parents)

        # crossover for offspring

        # mutate offspring

        # select survivors
