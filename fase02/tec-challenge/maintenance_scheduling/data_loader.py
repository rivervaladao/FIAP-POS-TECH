# data_loader.py
import pandas as pd
from worker import Worker
from task import MaintenanceTask
from operation_task import OperationTask
from datetime import datetime
from util import calculate_end_time

def load_workers_from_csv(disponibilidade_file, historico_file):
    disponibilidade_df = pd.read_csv(disponibilidade_file)
    historico_df = pd.read_csv(historico_file)

    workers = []
    for row in disponibilidade_df.itertuples(index=False):
        experiencia = historico_df[historico_df['matricula'] == row.matricula]['equipamento'].value_counts().to_dict()
        worker = Worker(
            worker_id=row.matricula,
            skills=str(row.qualificacao).split('/') if pd.notna(row.qualificacao) else [],  # Handle NaN or None
            experience_with_assets=experiencia,
            total_hours = (lambda time_str: sum(int(x) * 60 ** i for i, x in enumerate(reversed(time_str.split(':')))))(row.hora_total) if pd.notna(row.hora_total) else 0            
        )
        workers.append(worker)
    return workers

def load_tasks_from_csv(ordens_file):
    """
    Carrega ordens de serviço e suas operações a partir de um arquivo CSV.
    Para cada ordem, as operações são adicionadas e suas horas de início são ajustadas
    com base na hora de término da operação anterior.
    """
    ordens_df = pd.read_csv(ordens_file)
    tasks = {}
    for row in ordens_df.itertuples(index=False):
        ordem_id = row.ordem
        if ordem_id not in tasks:
            task = MaintenanceTask(
                task_id=row.ordem,
                due_date=datetime.strptime(row.data_inicio_base, '%d/%m/%Y').date(),
                start_time = row.hora_inicio_base,
                priority=row.indice_irpe
            )
            tasks[ordem_id] = task
        else:
            task = tasks[ordem_id]

        # Cria a operação dentro da ordem de serviço
        # Se for a primeira operação, usar a hora de início base da ordem
        if len(task.operations) == 0:
            op_effort = int(float(str(row.esforco_individual).replace(',','.'))*60) if pd.notna(row.esforco_individual) else 0 #em minutos 
            start_hour = row.hora_inicio_base if pd.notna(row.hora_inicio_base) else '00:00:01'
        else:
            # Hora de início da nova operação é a hora de término da última operação
            last_operation = task.operations[-1]
            start_hour = calculate_end_time(last_operation.start_hour, last_operation.effort)
        
        operation = OperationTask(
            operation_id=row.operacao,
            required_skill=str(row.qualificacao).split('/') if pd.notna(row.qualificacao) else [],  # Handle NaN or None
            due_date=datetime.strptime(row.data_inicio_base, '%d/%m/%Y').date(),
            asset=row.equipamento_ordem,
            effort= op_effort,
            start_hour=start_hour
        )
        
        # Adiciona a operação à ordem de serviço
        task.add_operation(operation)
    return list(tasks.values())
