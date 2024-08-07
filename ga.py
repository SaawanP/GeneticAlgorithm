import math

import crossover


class GA:
    def __init__(self,
                 num_generation,
                 population_size,
                 fitness_func,
                 crossover_func=crossover.single_point,
                 crossover_prob=1,
                 mutation_prob=0.05,
                 generational_gap=1,
                 parent_selection_type="fps",
                 on_fitness_calc=None,
                 on_parent_selection=None,
                 on_crossover=None,
                 on_mutate=None,
                 on_survivor_selection=None):

        self.current_generation = 0
        self.num_generation = num_generation
        self.population_size = population_size
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generational_gap = generational_gap

        # These will hold the best solution from every generation or the current best
        self.best_solution = []
        self.best_solution_fitness = []

        self.solutions = []
        self.solutions_fitness = []

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

        if callable(parent_selection_type) and parent_selection_type.__code__.co_argcount == 1:
            self.parent_select = parent_selection_type
        elif type(parent_selection_type) == str:
            if parent_selection_type == 'fps':
                self.parent_select = []
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
        pass

    def run(self):
        for i in range(self.num_generation):
            self.step()

    def step(self):
        if self.current_generation >= self.num_generation:
            print("Max number of generations reached, returning")
            return
        self.current_generation += 1

        # get fitness of population

        # get parents

        # crossover for offspring

        # mutate offspring

        # select survivors
