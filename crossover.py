import random
import numpy as np
import math


def single_point(parents, crossover_prob, genes_info):
    return N_point(parents, crossover_prob, genes_info, 1)


def N_point(parents, crossover_prob, genes_info, N):
    shape = [gene_info.get("length", 1) for gene_info in genes_info]

    if sum(shape) >= N - 1:
        return uniform_crossover(parents, crossover_prob, genes_info)

    children = []
    for i in range(len(parents) - 1):
        child1, child2, complete = _get_children(parents, crossover_prob, i)
        if not complete:
            children.append(child1)
            continue

        crossover_points = sorted(random.sample(range(1, sum(shape)), N))

        k = 0
        s = shape[k]
        for point in crossover_points:
            if point > s:
                k += 1
                s += shape[k]
            temp = child1[:]
            child1[k:] = child2[k:]
            child2[k:] = temp[k:]
            gene1 = child1[k]
            gene2 = child2[k]

            child1[k], child2[k] = _swap_gene(gene1, gene2, genes_info[k], s - point)

        children.append(child1)
        children.append(child2)

    return children


def uniform_crossover(parents, crossover_prob, genes_info):
    shape = [gene_info.get("length", 1) for gene_info in genes_info]

    children = []
    for i in range(len(parents) - 1):
        child1, child2, complete = _get_children(parents, crossover_prob, i)
        if not complete:
            children.append(child1)
            continue

        k = 0
        s = shape[k]
        for j in range(sum(shape)):
            if j > s:
                k += 1
                s += shape[k]

            gene1 = child1[j]
            gene2 = child2[j]
            child1[k], child2[k] = _swap_gene(gene1, gene2, genes_info[k], s - j)

        children.append(child1)
        children.append(child2)

    return children


def _swap_gene(gene1, gene2, gene_info, i):
    if gene_info["type"] == "binary" or gene_info["type"] == "discrete":
        gene1 = bin(gene1)[2:].zfill(gene_info["length"])
        gene2 = bin(gene2)[2:].zfill(gene_info["length"])

        temp = gene1
        gene1 = gene1[:i] + gene2[i:]
        gene2 = gene2[:i] + temp[i:]

        gene1 = int(gene1, 2)
        gene2 = int(gene2, 2)

        if gene_info["type"] == "discrete":  # check if new mutation in within range, otherwise clamp it
            if gene1 * gene_info["precision"] > abs(gene_info["range"][1] - gene_info["range"][0]):
                gene1 = int(gene_info["range"][1] - gene_info["range"][0])
        if gene_info["type"] == "discrete":
            if gene2 * gene_info["precision"] > abs(gene_info["range"][1] - gene_info["range"][0]):
                gene2 = int(gene_info["range"][1] - gene_info["range"][0])
    elif gene_info["type"] == "real" or gene_info["type"] == "integer":
        alpha = 0.2
        gene1 = alpha * gene1 + (1 - alpha) * gene2
        gene2 = alpha * gene2 + (1 - alpha) * gene1

        if gene_info["type"] == "integer":
            gene1 = math.floor(gene1)
            gene2 = math.ceil(gene2)
    return gene1, gene2


def _get_children(parents, crossover_prob, i):
    if crossover_prob is not None:
        indices = [i]
        while len(indices) == 1 and indices[0] == i:
            probs = np.random.random(size=len(parents))
            indices = list(set(np.where(probs <= crossover_prob)[0]))

        if len(indices) == 0:
            return parents[i], (), False

        idx = random.choice(indices)
        while idx != i:
            idx = random.choice(indices)

        child1 = parents[i]
        child2 = parents[idx]

    else:
        child1 = parents[i]
        child2 = parents[i + 1]

    return child1, child2, True
