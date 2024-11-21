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
        """
        Calcula a pontuação de fitness de um indivíduo (solução).
        O objetivo é maximizar a eficiência do planejamento, garantindo que as operações sejam executadas por trabalhadores
        qualificados, que as ordens de prioridade sejam realizadas primeiro e que o tempo de execução seja minimizado.
        """
        score = 0
        
        # Critérios de fitness
        for task in individual:
            for operation in task.operations:
                # Critério 1: Verificar se os trabalhadores alocados têm as qualificações necessárias
                for worker in operation.allocated_workers:
                    if worker.has_skill(operation.required_skill):
                        score += 10  # Aumenta a pontuação para trabalhadores qualificados

                # Critério 2: Priorização de ordens com maior índice de prioridade (indice_irpe)
                score += task.priority  # Prioridade alta aumenta o score

                # Critério 3: Penalizar se o tempo de execução da operação excede um limite
                # Penalize se a operação durar muito tempo (por exemplo, mais de 8 horas por dia)
                if operation.effort > 8:
                    score -= 5  # Penalizar por operações muito longas
                
                # Critério 4: Minimizar o tempo total de execução (quanto menor o esforço, melhor)
                score -= operation.effort  # Penaliza por operações mais longas
                
                # Critério 5: Recompensa por otimizar o uso da disponibilidade dos trabalhadores
                for worker in operation.allocated_workers:
                    if worker.is_available(operation.due_date, operation.start_hour, operation.effort):
                        score += 5  # Recompensa por um bom uso da disponibilidade

        return score
    
    def selection(self, population):
        """
        Realiza um torneio de seleção para escolher os melhores indivíduos.
        """
        selected = []
        tournament_size = 5
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            best_individual = max(tournament, key=lambda ind: self.fitness(ind))
            selected.append(best_individual)
        return selected

    def crossover(self, parent1, parent2):
        """
        Executa o crossover entre dois pais, combinando diferentes partes de ambos para gerar um filho mais diversificado.
        """
        point1 = random.randint(0, len(self.tasks) // 2)
        point2 = random.randint(point1, len(self.tasks) - 1)
        
        # Combinação de genes dos pais em diferentes intervalos
        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        return child

    def mutate(self, individual):
        """
        Realiza a mutação em um indivíduo para introduzir variação. A mutação troca dois genes (operações).
        """
        if random.random() < self.mutation_rate:
            idx1 = random.randint(0, len(individual) - 1)
            idx2 = random.randint(0, len(individual) - 1)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]  # Troca os genes
        return individual

    def evolve(self, population):
        """
        Evolui a população atual para gerar a próxima geração.
        """
        new_population = []
        selected_individuals = self.selection(population)
        
        for _ in range(0, len(population), 2):
            parent1 = random.choice(selected_individuals)
            parent2 = random.choice(selected_individuals)
            
            # Crossover entre os pais
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            
            # Mutação
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            new_population.append(child2)
        
        return new_population

    def optimize(self):
        population = self.initial_population()
        for generation in range(self.generations):
            population = self.evolve(population)
            best_individual = max(population, key=lambda ind: self.fitness(ind))
            print(f"Generation {generation} | Best Fitness: {self.fitness(best_individual)}")
        return best_individual
