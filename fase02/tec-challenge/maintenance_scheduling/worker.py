# worker.py
from typing import List
from datetime import datetime, date
from util import calculate_end_time
class Worker:
    """
    Representa um colaborador com habilidades, disponibilidade e experiência.

    Attributes:
        worker_id (str): ID do colaborador.
        skills (list): Lista de habilidades do colaborador.
        experience_with_assets (dict): Mapeia o ativo e o número de execuções no equipamento.
        hours_allocated (dict): Mapeia as datas e o tempo alocado para o colaborador.
        total_hours (int): tempo total disponíveis para alocação em minutos.
    """
    def __init__(self, worker_id, skills: List[str], experience_with_assets: dict, total_hours: float):
        self.worker_id = worker_id
        self.skills = skills
        self.experience_with_assets = experience_with_assets
        self.hours_allocated = {}
        self.total_hours = total_hours
        self.operations = [] 

    def is_available(self, date, start_time, effort:int):
        """
        Verifica se o colaborador está disponível na data e hora específicas com horas suficientes,
        e se não há sobreposição de operações.
        """        
        # Verifica se a nova operação sobrepõe operações já alocadas
        for operation in self.operations:
            if operation.due_date == date:
                current_end_time = calculate_end_time(operation.start_hour, operation.effort)
                new_end_time = calculate_end_time(start_time, effort)
                
                # Verificar se as operações se sobrepõem
                if (start_time < current_end_time and new_end_time > operation.start_hour):
                    return False
        
        # Soma o tempo já alocado nessa data
        horas_alocadas = self.hours_allocated.get(date, 0)
        
        # Verifica se o colaborador tem horas suficientes para o esforço adicional
        available_hours = self.total_hours - horas_alocadas
        return available_hours >= effort

    def has_skill(self, required_skills):
        """
        Verifica se o colaborador possui alguma das qualificações necessárias.
        required_skills pode ser uma string ou uma lista de qualificações.
        """
        if isinstance(required_skills, str):
            return required_skills in self.skills
        elif isinstance(required_skills, list):
            return any(skill in self.skills for skill in required_skills)

    def allocate_hours(self, operation):
        """
        Aloca horas e registra a operação para o colaborador na data específica.
        """
        if operation.due_date not in self.hours_allocated:
            self.hours_allocated[operation.due_date] = 0
        
        if self.is_available(operation.due_date, operation.start_hour, operation.effort):
            self.hours_allocated[operation.due_date] += operation.effort
            self.operations.append(operation)
        else:
            print (f"Colaborador {self.worker_id} não pode ser alocado para a operação {operation.operation_id} devido a sobreposição de horário {operation.due_date} {operation.start_hour}.")
            #raise ValueError(f"Colaborador {self.worker_id} não pode ser alocado para a operação devido a sobreposição de horário.")