# operation_task.py
from typing import List
from datetime import datetime, date
from worker import Worker

class OperationTask:
    """
    Representa uma operação dentro de uma ordem de serviço.

    Attributes:
        operation_id (str): ID da operação.
        required_skill (list): Habilidade necessária para realizar a operação.
        due_date (str): Data prevista para execução.
        asset (str): Ativo relacionado à operação.
        effort (float): Esforço necessário para executar a operação.
        start_hour (str): Hora de início da operação.
        allocated_workers (list): Lista de colaboradores alocados para essa operação.
    """
    def __init__(self, operation_id, required_skill: List[str], due_date: date, asset: str, effort: int, start_hour):
        self.operation_id = operation_id
        self.required_skill = required_skill
        self.due_date: date = due_date
        self.asset = asset
        self.effort:int = effort
        self.start_hour = start_hour
        self.allocated_workers = []
    
    def assign_worker(self, worker: Worker):
        """
        Aloca um colaborador para a operação.
        """
        self.allocated_workers.append(worker)
        worker.allocate_hours(self)