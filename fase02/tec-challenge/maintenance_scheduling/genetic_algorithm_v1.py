# genetic_algorithm.py
import random
from typing import List
from task import MaintenanceTask
from worker import Worker

class GeneticAlgorithm:
    """
    Implementa o Algoritmo Genético para otimizar o planejamento de manutenção.
    """

    def __init__(self, tasks: List[MaintenanceTask], workers: List[Worker], population_size=50, generations=50, mutation_rate=0.05):
        self.tasks = tasks
        self.workers = workers
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = []
            for task in self.tasks:
                task.assign_workers_to_operations(self.workers)
                individual.append(task)
            population.append(individual)
        return population

    def fitness(self, individual):
        score = 0
        for task in individual:
            for operation in task.operations:
                for worker in operation.allocated_workers:                        
                    # Peso para habilidades compatíveis
                    if worker.has_skill(operation.required_skill):
                        score += 3                    
                    # Peso maior para experiência no ativo
                    if worker.experience_with_assets.get(operation.asset, 0) > 0:
                        score += 2
                    # Verificar disponibilidade
                    if worker.is_available(date=operation.due_date,start_time=operation.start_hour,effort=operation.effort):
                        score += 1
        
        return score
       
    def selection(self, population):
        population = sorted(population, key=lambda ind: self.fitness(ind), reverse=True)
        return population[:int(len(population) / 2)]

    def crossover(self, parent1, parent2):
        point = random.randint(0, len(self.tasks) - 1)
        child = parent1[:point] + parent2[point:]
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(individual) - 1)
            task = individual[index]
            task.assign_workers_to_operations(self.workers)
        return individual

    def evolve(self, population):
        new_population = []
        selected_individuals = self.selection(population)
        for _ in range(len(population)):
            parent1 = random.choice(selected_individuals)
            parent2 = random.choice(selected_individuals)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def optimize(self):
        population = self.initial_population()
        for generation in range(self.generations):
            population = self.evolve(population)
            best_individual = max(population, key=lambda ind: self.fitness(ind))
            print(f"Generation {generation} | Best Fitness: {self.fitness(best_individual)}")
        return best_individual
