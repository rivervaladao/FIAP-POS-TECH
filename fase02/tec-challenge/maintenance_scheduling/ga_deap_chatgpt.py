import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from datetime import datetime, time

# Carregar os dados
ordens_df = pd.read_csv("./data/ordens_manutencao.csv")
disponibilidade_df = pd.read_csv("./data/disponibilidade_full.csv")
historico_df = pd.read_csv("./data/historico_manutencao.csv")

# Função para limpar e padronizar colunas de horário
def clean_time_column(time_value):
    if isinstance(time_value, time):  # Se já for do tipo time, retorna como está
        return time_value
    try:
        # Padronizar para '%H:%M' se possível
        return datetime.strptime(time_value.strip(), '%H:%M').time()
    except (ValueError, AttributeError):
        # Substituir horários inválidos ou None por um valor padrão (por exemplo, '00:00')
        return time(0, 0)  # Usa diretamente a classe time para criar 00:00

# Pré-processar os dados
def preprocess_data():
    # Preencher valores NaN em quantidade_executantes e garantir tipo inteiro
    ordens_df["quantidade_executantes"] = ordens_df["quantidade_executantes"].fillna(1).astype(int)
    
    # Padronizar horários no DataFrame de disponibilidade
    disponibilidade_df["hora_inicio"] = disponibilidade_df["hora_inicio"].fillna("00:00").apply(clean_time_column)
    disponibilidade_df["hora_fim"] = disponibilidade_df["hora_fim"].fillna("00:00").apply(clean_time_column)
    # Garantir que 'hora_total' em disponibilidade seja numérico
    disponibilidade_df["hora_total"] = pd.to_numeric(disponibilidade_df["hora_total"], errors="coerce").fillna(0)
    
    # Padronizar horários no DataFrame de ordens
    ordens_df["hora_inicio_base"] = ordens_df["hora_inicio_base"].fillna("00:00").apply(clean_time_column)
    # Garantir que 'esforco_individual' seja numérico, substituir NaN por 0
    ordens_df["esforco_individual"] = pd.to_numeric(ordens_df["esforco_individual"], errors="coerce").fillna(0)

    
    # Ordenar as ordens pelo índice de prioridade
    ordens_df.sort_values(by="indice_irpe", ascending=False, inplace=True)
    
    # Processar qualificações para facilitar comparação
    disponibilidade_df["qualificacao"] = disponibilidade_df["qualificacao"].fillna("").apply(lambda x: x.split("/") if isinstance(x, str) else [])
    ordens_df["qualificacao"] = ordens_df["qualificacao"].fillna("").apply(lambda x: x.split("/") if isinstance(x, str) else [])
    
    return ordens_df, disponibilidade_df, historico_df

ordens_df, disponibilidade_df, historico_df = preprocess_data()

# Representação e configuração do DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Indivíduo representa uma solução: [ordem_id, operação, [funcionários alocados], horário de início]
def generate_individual():
    individual = []
    for _, ordem in ordens_df.iterrows():
        operacoes = [ordem["operacao"]] * ordem["quantidade_executantes"]
        alocacoes = []
        for operacao in operacoes:
            alocacao = [
                ordem["ordem"],
                operacao,
                random.choice(disponibilidade_df["matricula"].tolist()),  # Escolha aleatória inicial
                ordem["data_inicio_base"],
                ordem["hora_inicio_base"]
            ]
            alocacoes.append(alocacao)
        individual.extend(alocacoes)
    return individual

toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Função de avaliação
def evaluate(individual):
    penalty = 0
    
    for allocation in individual:
        ordem_id, operacao, matricula, data_inicio, hora_inicio = allocation
        ordem = ordens_df[ordens_df["ordem"] == ordem_id].iloc[0]
        
        # Validar horários
        hora_inicio = clean_time_column(hora_inicio)
        funcionario = disponibilidade_df[disponibilidade_df["matricula"] == matricula].iloc[0]
        inicio_turno = funcionario["hora_inicio"]
        fim_turno = funcionario["hora_fim"]
        
        # Restrição de qualificação
        qualificacoes_necessarias = ordem["qualificacao"]
        qualificacoes_funcionario = funcionario["qualificacao"]  # Agora garantido ser uma lista
        if not set(qualificacoes_necessarias).issubset(set(qualificacoes_funcionario)):
            penalty += 10  # Penalidade por qualificação
        
        # Restrição de horas disponíveis
        esforco_individual = float(ordem["esforco_individual"])  # Garantir que é float
        hora_total_disponivel = float(funcionario["hora_total"])  # Garantir que é float
        if esforco_individual > hora_total_disponivel:
            penalty += 20  # Penalidade por falta de horas disponíveis
        
        # Restrição de turno
        if not (inicio_turno <= hora_inicio <= fim_turno):
            penalty += 15  # Penalidade por alocação fora do turno
    
    return (penalty,)

toolbox.register("evaluate", evaluate)

# Operadores Genéticos
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Executar o Algoritmo Genético
def run_genetic_algorithm():
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, 
                        stats=None, halloffame=hof, verbose=True)
    
    return hof[0]

best_schedule = run_genetic_algorithm()

# Apresentar os Resultados
result_df = pd.DataFrame(best_schedule, columns=["ordem", "operacao", "matriculas", "data_inicio_base", "horario_alocado"])
result_df.to_csv("./data/resultados_escala.csv", index=False)
print("Escalonamento gerado e salvo em './data/resultados_escala.csv'.")
