from sklearn.pipeline import Pipeline
from trade.Stock_Analyzer import *
from trade.StockProcessing import *
from trade.Accuracy_Analyzer import *

class CustomPipeline:
    def __init__(self):
        # Создание конвейера с указанными шагами
        self.pipeline = Pipeline(steps=[
            ('RSI', RSIAnalyzer()),  # добавляем признаки из классического RSI индикатора
            # ('Debugger1', Debugger()),  # Проверка после RSIAnalyzer
            ('SMA', SMAAnalyzer()),
            # ('Debugger2', Debugger()),  # Проверка после SMAAnalyzer
            ('EMA', EMAAnalyzer()),
            ('RSIEMA', RSIEMAAnalyzer()),
            ('ROC', ROCAnalyzer()),
            ('MOM', MomentumAnalyzer()),
            ('STOCH', StochasticOscillatorAnalyzer()),
            ('SHIFT1', ShiftedColumnAdder(columns_to_shift=['open', 'close', 'high', 'low', 'volume'])),
            ('SHIFT2', ShiftedColumnAdder(num_shifts=1, columns_to_shift=['EMA_short', 'EMA_long', 'EMA_longest'])),
            ('TIME_CONVERTER', DateTimeConverter()),
            ('DEPENDENCIES_REMOVER', DependentFeatureRemover()),  
            ('CORRELATION_REMOVER', LowCorrelationRemover(target_column='result')),
            ('DATAPREPROCESSOR', DataPreprocessor(target_column='result',
                                                  apply_scaling=True,
                                                  apply_balancing=False))  # чтобы не перемешивать последовательности до разделения на обучающую и тестовую последовательности
        ])

    def get_pipeline(self):
        """Метод для получения сформированного конвейера."""
        return self.pipeline

# Пример использования класса
#if __name__ == "__main__":
#    custom_pipeline = CustomPipeline()
#    pipeline_transform = custom_pipeline.get_pipeline()
    

