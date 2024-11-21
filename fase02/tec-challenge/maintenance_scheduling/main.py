# main.py
from genetic_algorithm_v2 import GeneticAlgorithm
from data_loader import load_workers_from_csv, load_tasks_from_csv
from util import calculate_end_time

# Carregar os dados
historico_manutencao_csv = 'data/historico_manutencao.csv'
disponibilidade_csv = 'data/disponibilidade_full.csv'
ordens_manutencao_csv = 'data/ordens_manutencao.csv'


workers = load_workers_from_csv(disponibilidade_csv, historico_manutencao_csv)
tasks = load_tasks_from_csv(ordens_manutencao_csv)

# Instanciando o Algoritmo Genético
genetic_algo = GeneticAlgorithm(tasks, workers, population_size=10, generations=10, mutation_rate=0.05)

# Executar a otimização
best_solution = genetic_algo.optimize()

# Exibir o melhor planejamento, evitando sobreposições
print("Ordem | Lista de Operações | Lista de Colaboradores | Data Início | Hora Início da Operação | Hora de Término")
for task in best_solution:
    operation_details = []
    for operation in sorted(task.operations, key=lambda op: op.operation_id):
        colaboradores = ', '.join([str(worker.worker_id) for worker in operation.allocated_workers])
        hora_termino = calculate_end_time(operation.start_hour, operation.effort)
        #operation_details.append(f"Operação {operation.operation_id} - Colaboradores: {colaboradores} - Hora Início: {operation.start_hour} - Término: {hora_termino}")    
        print(f"{task.task_id} | {operation.operation_id} | {colaboradores}| {task.due_date} | {operation.start_hour} | {hora_termino} ")