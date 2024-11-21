from datetime import datetime, timedelta

def calculate_end_time(start_hour, effort):
    """
    Calcula a hora de término de uma operação com base na hora de início e na duração (esforço) em horas.
    """
    hour_format = "%H:%M:%S"  # Formato para a hora (HH:MM)
    init_hour_dt = datetime.strptime(start_hour, hour_format)
    end_hour_dt = init_hour_dt + timedelta(minutes=effort)
    return end_hour_dt.strftime(hour_format)
