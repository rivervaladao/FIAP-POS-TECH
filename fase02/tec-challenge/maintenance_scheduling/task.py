# task.py
from typing import List
from datetime import date, datetime, timedelta
from operation_task import OperationTask
from worker import Worker
from util import calculate_end_time
class MaintenanceTask:
    """
    Representa uma tarefa de manutenção.

    Attributes:
        task_id (str): ID da ordem de serviço.
        operations (list): Lista de operações dentro da ordem de serviço.
        due_date (str): Data de início da ordem de serviço.
        start_hour (str): Hora de início da primeira operação.
    """

    def __init__(self, task_id, due_date: date, start_time, priority):
        self.task_id = task_id
        self.due_date: date = due_date
        self.start_hour = start_time
        self.priority = priority
        self.operations: List[OperationTask] = []

    def add_operation(self, operation_task: OperationTask):
        """
        Adiciona uma operação à ordem de serviço.
        """
        self.operations.append(operation_task)
    
    def assign_workers_to_operations(self, workers:List[Worker]):
        """
        Aloca trabalhadores para as operações, garantindo que eles possuam as qualificações necessárias.
        Também ajusta a hora de início se necessário para evitar sobreposição.
        """
        for operation in sorted(self.operations, key=lambda op: op.operation_id):
            # Filtrar os trabalhadores qualificados para a operação
            qualified_workers = [worker for worker in workers if worker.has_skill(operation.required_skill)]
            
            # Se nenhum colaborador for qualificado, selecionar por experiência no ativo
            if not qualified_workers:
                qualified_workers = sorted(workers, key=lambda w: w.experience_with_assets.get(operation.asset, 0), reverse=True)
            
            for worker in qualified_workers:
                if worker.is_available(operation.due_date, operation.start_hour, operation.effort):
                    # Se disponível e qualificado, aloca o colaborador
                    worker.allocate_hours(operation)
                    operation.assign_worker(worker)
                    break
                else:
                    # Se houver sobreposição, ajustar a hora de início
                    last_end_hour = max([calculate_end_time(op.start_hour, op.effort) for op in worker.operations if op.due_date == operation.due_date], default=operation.start_hour)
                    new_start_hour = last_end_hour
                    if worker.is_available(operation.due_date, new_start_hour, operation.effort):
                        operation.start_hour = new_start_hour  # Ajustar a hora de início
                        worker.allocate_hours(operation)
                        operation.assign_worker(worker)
                        break

    def calculate_total_duration(self):
            """
            Calcula a duração acumulada da ordem (soma de todas as operações).
            """
            total_duration = sum(operation.effort for operation in self.operations)
            return total_duration

    def calculate_end_time(self):
        """
        Calcula a hora de término da ordem somando o esforço de cada operação à hora de início base.
        """
        total_duration = self.calculate_total_duration()
        formato_hora = "%H:%M"
        hora_inicio_dt = datetime.strptime(self.hora_inicio_base, formato_hora)
        hora_termino_dt = hora_inicio_dt + timedelta(hours=total_duration)
        return hora_termino_dt.strftime(formato_hora)                    