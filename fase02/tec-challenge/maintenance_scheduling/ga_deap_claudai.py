import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from datetime import datetime, timedelta
import itertools
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

class TurnScheduling:
    def __init__(self, ordens_path, disponibilidade_path, historico_path):
        # Carregar arquivos CSV
        self.ordens = pd.read_csv(ordens_path)
        self.disponibilidade = pd.read_csv(disponibilidade_path)
        self.historico = pd.read_csv(historico_path)
        
        # Preparar dados
        self.prepare_data()
        
        # Configurar DEAP
        self.setup_deap()
    
    def prepare_data(self):
        # Garantir que colunas de qualificação sejam tratadas como strings
        self.ordens['qualificacao'] = self.ordens['qualificacao'].fillna('').astype(str)
        # Garantir que quantidade_executantes seja um inteiro
        self.ordens['quantidade_executantes'] = pd.to_numeric(self.ordens['quantidade_executantes'], errors='coerce').fillna(1).astype(int)
        
        # Calcular o número total de executores necessários
        self.total_executores = int(self.ordens['quantidade_executantes'].sum())       
        self.disponibilidade['qualificacao'] = self.disponibilidade['qualificacao'].fillna('').astype(str)
        
        # Converter colunas para string
        self.ordens = self.ordens.astype(str)
        self.disponibilidade = self.disponibilidade.astype(str)
        self.historico = self.historico.astype(str)
        
        # Ordenar ordens por índice IRPE (prioridade)
        try:
            self.ordens['indice_irpe'] = pd.to_numeric(self.ordens['indice_irpe'], errors='coerce').fillna(0)
            self.ordens = self.ordens.sort_values('indice_irpe', ascending=False)
        except Exception as e:
            logging.warning(f"Erro ao ordenar ordens por IRPE: {e}")
        
        # Criar dicionário de qualificações por funcionário
        self.employee_qualifications = {}
        for _, row in self.disponibilidade.iterrows():
            matricula = row['matricula']
            qualificacoes = row['qualificacao'].split('/') if row['qualificacao'] else []
            self.employee_qualifications[matricula] = set(qualificacoes)
        
        # Criar histórico de manutenção por equipamento
        self.equipment_maintenance_history = {}
        for _, row in self.historico.iterrows():
            key = row['equipamento']
            if key not in self.equipment_maintenance_history:
                self.equipment_maintenance_history[key] = []
            self.equipment_maintenance_history[key].append(row['matricula'])

        # Adicionar logs de preparação
        logging.info(f"Total de ordens processadas: {len(self.ordens)}")
        logging.info(f"Total de funcionários: {len(self.disponibilidade)}")
        logging.info(f"Total de registros de histórico: {len(self.historico)}")

    def setup_deap(self):
        # Destruir criadores existentes para evitar erros de re-registro
        try:
            del creator.FitnessMin
            del creator.Individual
        except Exception:
            pass
        
        # Configurar framework DEAP para resolução do problema
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Criar toolbox
        self.toolbox = base.Toolbox()
        
        # Definir operadores genéticos
        self.toolbox.register("attr_employee", self.select_employee)
        # Usar self.total_executores calculado em prepare_data
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                               self.toolbox.attr_employee, n=self.total_executores)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Registrar funções de avaliação e seleção
        self.toolbox.register("evaluate", self.evaluate_schedule)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_schedule)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def select_employee(self):
        valid_employees = []
        for _, ordem in self.ordens.iterrows():
            centro_trabalho = str(ordem['centro_trabalho'])
            qualificacoes = set(ordem['qualificacao'].split('/') if ordem['qualificacao'] else [])
            quantidade_executantes = int(ordem['quantidade_executantes'])
            
            qualified_employees = [
                emp for emp, quals in self.employee_qualifications.items()
                if self.disponibilidade.loc[self.disponibilidade['matricula'] == emp, 'centro_trabalho'].iloc[0] == centro_trabalho
                and (not qualificacoes or qualificacoes.issubset(quals))
            ]
            
            if not qualified_employees and 'equipamento_ordem' in ordem:
                equip_ordem = ordem['equipamento_ordem']
                qualified_employees = self.equipment_maintenance_history.get(equip_ordem, [])
            
            employees_to_use = qualified_employees or self.disponibilidade[
                self.disponibilidade['centro_trabalho'] == centro_trabalho
            ]['matricula'].tolist()
            
            selected_employees = random.choices(employees_to_use, k=quantidade_executantes) if employees_to_use else [''] * quantidade_executantes
            valid_employees.extend(selected_employees)
        
        return valid_employees
    
    def evaluate_schedule(self, individual):
        penalties = 0
        index = 0
        
        for _, ordem in self.ordens.iterrows():
            quantidade_executantes = int(ordem['quantidade_executantes'])
            employees = individual[index:index + quantidade_executantes]
            
            for employee in employees:
                if not isinstance(employee, str):
                    logging.warning(f"Tipo inválido para funcionário: {type(employee)}")
                    penalties += 200
                    continue

                if not employee:
                    penalties += 200
                    continue
                
                if not self.check_qualifications(employee, ordem):
                    penalties += 100
                
                if not self.check_time_availability(employee, ordem):
                    penalties += 50
                
                if not self.check_total_allocated_time(employee, ordem):
                    penalties += 25
            
            index += quantidade_executantes
        
        return (penalties,)
    
    def check_qualifications(self, employee, ordem):
        if not employee or not isinstance(employee, str):
            logging.warning(f"Invalid employee type: {employee} (type: {type(employee)})")
            return False
        
        emp_qualifications = self.employee_qualifications.get(employee, set())
        req_qualifications = set(ordem.qualificacao.split('/') if ordem.qualificacao else [])
        return not req_qualifications or req_qualifications.issubset(emp_qualifications)

    def check_time_availability(self, employee, ordem):
        # Skip validation if no employee is assigned
        if not employee:
            return False
        
        # Verificar se o horário da ordem respeita o turno do funcionário
        try:
            emp_data = self.disponibilidade[self.disponibilidade['matricula'] == employee].iloc[0]
            ordem_start = datetime.strptime(f"{ordem.data_inicio_base} {ordem.hora_inicio_base}", "%Y-%m-%d %H:%M")
            return (datetime.strptime(emp_data['hora_inicio'], "%H:%M") <= ordem_start and
                    datetime.strptime(emp_data['hora_fim'], "%H:%M") >= ordem_start)
        except Exception:
            return False

    def check_total_allocated_time(self, employee, ordem):
        # Skip validation if no employee is assigned
        if not employee:
            return False
        
        # Verificar tempo total alocado do funcionário
        try:
            emp_data = self.disponibilidade[self.disponibilidade['matricula'] == employee].iloc[0]
            return emp_data['hora_total'] >= ordem.esforco_individual
        except Exception:
            return False
    
    def mutate_schedule(self, individual, indpb=0.1):
        index = 0
        for _, ordem in self.ordens.iterrows():
            quantidade_executantes = int(ordem['quantidade_executantes'])
            for i in range(index, index + quantidade_executantes):
                if random.random() < indpb:
                    valid_employees = self.disponibilidade[
                        (self.disponibilidade['centro_trabalho'] == ordem['centro_trabalho'])
                    ]['matricula'].tolist()
                    individual[i] = random.choice(valid_employees or [''])
            index += quantidade_executantes
        return (individual,)

    def solve(self, population_size=100, generations=50):
        logging.info(f"Iniciando otimização com {population_size} indivíduos e {generations} gerações")
        
        population = self.toolbox.population(n=population_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for gen in range(generations):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.7, mutpb=0.2)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select the next generation population
            population[:] = self.toolbox.select(offspring, k=len(population))
            
            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            logging.info(f"Geração {gen}: {record}")

        best_schedule = tools.selBest(population, k=1)[0]
        logging.info(f"Melhor solução encontrada com fitness: {best_schedule.fitness.values[0]}")
        
        return self.format_schedule(best_schedule)

    def format_schedule(self, solution):
        schedule_results = []
        index = 0
        for _, ordem in self.ordens.iterrows():
            quantidade_executantes = int(ordem['quantidade_executantes'])
            employees = solution[index:index + quantidade_executantes]
            for employee in employees:
                schedule_results.append({
                    'ordem': ordem['ordem'],
                    'operacao': ordem['operacao'],
                    'matricula': employee,
                    'data_inicio_base': ordem['data_inicio_base'],
                    'horario_alocado': ordem['hora_inicio_base']
                })
            index += quantidade_executantes
        
        return pd.DataFrame(schedule_results)

# Exemplo de uso
def main():
    try:
        scheduler = TurnScheduling(
            './data/ordens_manutencao.csv', 
            './data/disponibilidade_full.csv', 
            './data/historico_manutencao.csv'
        )
        
        resultado_escalonamento = scheduler.solve()
        resultado_escalonamento.to_csv('resultado_escalonamento.csv', index=False)
        print(resultado_escalonamento)
    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()  # Adiciona rastreamento completo do erro

if __name__ == "__main__":
    main()