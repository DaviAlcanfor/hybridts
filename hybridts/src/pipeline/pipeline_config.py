from dataclasses import dataclass
from typing import Optional
from datetime import date
import calendar
from dateutil.relativedelta import relativedelta

@dataclass
class PipelineConfig:
    """
    Configuration for forecast pipeline execution.
    
    Manages forecast horizon calculation based on target month (current or next),
    and provides utilities for date-based pipeline parameters.
    
    Attributes:
        modelo: Model choice identifier (e.g., "xgboost", "lightgbm")
        target: Target metric name (e.g., "revenue", "demand")
        horizon_str: Horizon specification - either number of days as string or "Nenhum" for auto
        mes: Target month name for forecast
        nome_mes_atual: Current month name
        nome_mes_prox: Next month name
        hoje: Reference date (usually today)
        horizon_dias: Calculated forecast horizon in days (auto-computed)
    """
    modelo: str
    target: str
    horizon_str: str
    mes: str
    nome_mes_atual: str              
    nome_mes_prox: str               
    hoje: date                      
    horizon_dias: Optional[int] = None

    def __post_init__(self):
        """
        Automatically calculate forecast horizon based on target month.
        
        If horizon_str is a number, uses it directly.
        If "Nenhum" (auto), calculates days remaining until end of target month.
        """
        
        if self.horizon_str != "Nenhum":
            self.horizon_dias = int(self.horizon_str)
            return
        
        # Se for esse mes, calcula os dias que faltam ate o fim do mes
        if self.mes == self.nome_mes_atual:
            ultimo_dia = calendar.monthrange(self.hoje.year, self.hoje.month)[1]
            data_fim = date(self.hoje.year, self.hoje.month, ultimo_dia)
            
        elif self.mes == self.nome_mes_prox:
            prox_mes_dt = self.hoje + relativedelta(months=1)
            ultimo_dia_prox = calendar.monthrange(prox_mes_dt.year, prox_mes_dt.month)[1]
            data_fim = date(prox_mes_dt.year, prox_mes_dt.month, ultimo_dia_prox)
        else:
            data_fim = self.hoje
        
        self.horizon_dias = (data_fim - self.hoje).days


    @property
    def mes_alvo_numero(self) -> int:
        """
        Get target month number (1-12).
        
        Returns:
            Month number of the forecast target period
        """
        if self.mes == self.nome_mes_prox:
             return (self.hoje + relativedelta(months=1)).month
        return self.hoje.month
    
    @property
    def ano_alvo_numero(self) -> int:
        """
        Get target year number.
        
        Returns:
            Year of the forecast target period
        """
        if self.mes == self.nome_mes_prox:
             return (self.hoje + relativedelta(months=1)).year
        return self.hoje.year