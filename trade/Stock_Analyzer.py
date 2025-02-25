import pandas as pd
import talib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column = 'begin'):
        self.datetime_column = datetime_column
        
    def fit(self, X, y=None):
        """
        Сохраняет DataFrame для дальнейшего использования.
        
        :param X: DataFrame с данными
        :param y: Целевые значения (не используются)
        :return: self
        """
        #self.X = X  # Сохраняем DataFrame для дальнейшего использования
        return self

    def transform(self, X):
        """
        Преобразует столбец с датами в новые столбцы.
        
        :return: Обновленный DataFrame с новыми признаками
        """
        
        # Извлечение компонентов даты
        X['year'] = X[self.datetime_column].dt.year
        X['month'] = X[self.datetime_column].dt.month
        X['day'] = X[self.datetime_column].dt.day
        X['day_of_week'] = X[self.datetime_column].dt.dayofweek  # 0 - понедельник, 6 - воскресенье
        
        # Разделение времени на диапазоны и удаление колонки datetime
        X['time_range'] = X[self.datetime_column].apply(self._get_time_range)
        X.drop(self.datetime_column, axis=1, inplace=True)  # Удаляем оригинальный столбец с датами
        
        return X  # Возвращаем обновленный DataFrame

    def _get_time_range(self, dt):
        """
        Определяет временной диапазон для заданной даты и времени.
        
        :param dt: объект datetime
        :return: номер временного диапазона (1, 2 или 3)
        """
        if dt.time() >= pd.to_datetime("10:00").time() and dt.time() < pd.to_datetime("14:00").time():
            return 1
        elif dt.time() >= pd.to_datetime("14:00").time() and dt.time() < pd.to_datetime("19:00").time():
            return 2
        else:
            return 3

class LowCorrelationRemover(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, threshold=0.03):
        """
        Инициализация класса CorrelationAnalyzer.

        :param target_column: Название целевой колонки для анализа корреляции.
        :param threshold: Порог для удаления колонок с низкой корреляцией.
        """
        self.target_column = target_column
        self.threshold = threshold
        self.drop_list_corr = []

    def fit(self, X, y=None):
        # В данном случае нет необходимости в обучении, поэтому просто возвращаем self
        return self

    def calculate_correlation(self, X):
        """Вычисляет корреляцию и возвращает матрицу корреляции."""
        return X.corr()

    def identify_low_correlation(self, correlation_matrix):
        """Идентифицирует колонки с корреляцией ниже заданного порога."""
        correlation_result = abs(correlation_matrix[self.target_column])
        self.drop_list_corr = sorted(list(correlation_result[correlation_result < self.threshold].index))

    def drop_low_correlation_columns(self, X):
        """Удаляет колонки с низкой корреляцией из датафрейма."""
        if self.drop_list_corr:
            #print("Columns dropped due to low correlation:", self.drop_list_corr)
            X = X.drop(labels=self.drop_list_corr, axis=1)
            return X
        else:
            print("No columns to drop.")
            return X

    def transform(self, X):
        """Выполняет полный анализ корреляции и удаляет колонки с низкой корреляцией."""
        correlation_matrix = self.calculate_correlation(X)
        self.identify_low_correlation(correlation_matrix)
        return self.drop_low_correlation_columns(X)

class DependentFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.9, target_column='result'):
        self.correlation_threshold = correlation_threshold
        self.dependent_features = set()
        self.target_column = target_column  # Название целевой колонки

    def fit(self, X, y=None):
        """
        Метод fit больше не нужен для вычисления зависимых признаков,
        но его можно оставить для совместимости с интерфейсом Scikit-learn.
        """
        return self

    def transform(self, X):
        """
        Вычисляет матрицу корреляций и удаляет зависимые признаки из DataFrame.

        :param X: Входной DataFrame.
        :return: DataFrame без зависимых признаков.
        """
        # Вычисляем матрицу корреляций
        corr_matrix = X.corr().abs()
        
        # Перебираем матрицу корреляций
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                # Сравниваем значения корреляции
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    colname = corr_matrix.columns[i]
                    self.dependent_features.add(colname)

        # Удаляем зависимые признаки из DataFrame, исключая целевую колонку
        if self.target_column in self.dependent_features:
            self.dependent_features.remove(self.target_column)

        # Удаляем зависимые признаки из DataFrame
        X = X.drop(columns=self.dependent_features)
        
        return X

    def get_dependent_features(self):
        """Возвращает набор зависимых признаков."""
        return self.dependent_features



class ShiftedColumnAdder(BaseEstimator, TransformerMixin):
    def __init__(self, num_shifts = 3, columns_to_shift=None):
        self.num_shifts = num_shifts
        self.columns_to_shift = columns_to_shift if columns_to_shift is not None else []

    def fit(self, X, y=None):
        # В данном случае нет необходимости в обучении, поэтому просто возвращаем self
        return self

    def add_shifted_columns_with_fill(self, X):
        X = X.bfill()  # Заполнение NaN значений вперед
        for name in self.columns_to_shift:
            for i in range(1, self.num_shifts + 1):
                X[f'Shifted_{name}_{i}'] = X[name].shift(i)
        X = X.bfill()  # Заполнение NaN значений вперед после сдвига
        return X

    def transform(self, X):
        return self.add_shifted_columns_with_fill(X)


class StochasticOscillatorAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, stoch_k=14, stoch_d=3, stoch_period=3):
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.stoch_period = stoch_period

    def fit(self, X, y=None):
        # В данном случае нет необходимости в обучении, поэтому просто возвращаем self
        return self

    def STOK(self, close, low, high, n):
        """Расчет %K стохастического осциллятора."""
        lowest_low = low.rolling(n).min()
        highest_high = high.rolling(n).max()
        
        # Проверка на деление на ноль
        denominator = highest_high - lowest_low
        
        # Устанавливаем %K в NaN, если знаменатель равен 0 или NaN
        STOK = ((close - lowest_low) / denominator).where(denominator != 0) * 100
        
        # Заполнение NaN значений
        #first_valid_value = STOK[STOK.first_valid_index()] if STOK.first_valid_index() is not None else 0
        STOK.fillna(method='bfill', inplace=True)  #fillna(first_valid_value, inplace=True)  # Заполнение первым ненулевым значением или 0
        return STOK

    def STOD(self, close, low, high, n, d_period):
        """Расчет %D стохастического осциллятора."""
        stok_values = self.STOK(close, low, high, n)
        
        # Сглаживание %K для получения %D
        STOD = stok_values.rolling(d_period).mean()
        
        # Заполнение NaN значений
        #first_valid_value = STOD[STOD.first_valid_index()] if STOD.first_valid_index() is not None else 0
        STOD.fillna(method='bfill', inplace=True)  #fillna(first_valid_value, inplace=True)  # Заполнение первым ненулевым значением или 0
        return STOD

    def f(self, k, d):
        """Функция для определения сигнала на основе %K и %D."""
        if k > d:
            return 1  # Сигнал на покупку
        elif k < d:
            return 0  # Сигнал на продажу
        else:
            return -1  # Нейтральный сигнал

    def transform(self, X):
        X['%K'] = self.STOK(X['close'], X['low'], X['high'], self.stoch_k)
        X['%D'] = self.STOD(X['close'], X['low'], X['high'], self.stoch_d, self.stoch_period)
        
        # Создание сигналов на основе %K и %D
        X['stoch_signal_1'] = X.apply(
            lambda x: 1 if (self.f(x['%K'], x['%D']) == 1 and x['%K'] <= 20 and x['%D'] <= 20) 
            else 0 if (self.f(x['%K'], x['%D']) == 0 and x['%K'] >= 80 and x['%D'] >= 80)
            else -1, axis=1)
        
        X['stoch_signal_2'] = X.apply(lambda x: self.f(x['%K'], x['%D']), axis=1)

        return X


class MomentumAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, mom_low=13, mom_high=21):
        self.mom_low = mom_low
        self.mom_high = mom_high

    def fit(self, X, y=None):
        # В данном случае нет необходимости в обучении, поэтому просто возвращаем self
        return self

    def MOM(self, df, n):
        MOM = df.diff(n)
        #first_valid_value = MOM[MOM.first_valid_index()]  # Находим первое ненулевое значение
        MOM.fillna(method='bfill', inplace=True)  #fillna(first_valid_value, inplace=True)  # Заполняем NaN
        return MOM

    def transform(self, X):
        # Рассчитываем импульс для заданных периодов
        X['mom_low'] = self.MOM(X['close'], self.mom_low)
        X['mom_high'] = self.MOM(X['close'], self.mom_high)

        # Добавляем колонки со сдвигом на 1
        X['mom_low_shifted'] = X['mom_low'].shift(1)
        X['mom_high_shifted'] = X['mom_high'].shift(1)

        # Заполнение NaN значений в новых колонках
        X['mom_low_shifted'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед
        X['mom_high_shifted'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед

        return X


class ROCAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, roc_low=9, roc_high=14):
        self.roc_low = roc_low
        self.roc_high = roc_high

    def fit(self, X, y=None):
        # В данном случае нет необходимости в обучении, поэтому просто возвращаем self
        return self

    def ROC(self, df, n):
        M = df.diff(n - 1)  
        N = df.shift(n - 1)  
        ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
        
        # Находим первое ненулевое значение и заполняем NaN
        #first_valid_value = ROC[ROC.first_valid_index()] 
        ROC.fillna(method='bfill', inplace=True)  #fillna(first_valid_value, inplace=True) 
        return ROC

    def transform(self, X):
        X['roc_low'] = self.ROC(X['close'], self.roc_low)
        X['roc_high'] = self.ROC(X['close'], self.roc_high)
        # Рассчитываем ROC для заданных периодов с сдвигом на 1
        X['roc_low_shifted'] = self.ROC(X['close'], self.roc_low).shift(1)  # Сдвиг на 1
        X['roc_high_shifted'] = self.ROC(X['close'], self.roc_high).shift(1)  # Сдвиг на 1
        
        # Заполнение NaN значений
        X['roc_low_shifted'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед
        X['roc_high_shifted'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед

        return X


class RSIEMAAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, period=21, h_bound=70):
        self.period = period
        self.h_bound = h_bound

    def fit(self, X, y=None):
        # В данном случае нет необходимости в обучении, поэтому просто возвращаем self
        return self

    def rsi_ema(self, prices):
        # Вычисляем изменения цен
        dta = prices.diff()

        # Положительные и отрицательные изменения
        gain = dta.clip(lower=0)  # Положительные изменения
        loss = -dta.clip(upper=0)  # Отрицательные изменения

        # Вычисляем экспоненциальное сглаженное среднее для прибыли и потерь
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        # Рассчитываем относительную силу (RS)
        rs = avg_gain / avg_loss.replace(0, np.nan)  # Заменяем 0 на NaN для предотвращения деления на ноль

        # Рассчитываем индекс относительной силы (RSI)
        rsi = 100 - (100 / (1 + rs))
        
        # Устанавливаем RSI в 100 там, где avg_loss был равен 0
        rsi[avg_loss == 0] = 100

        return rsi

    def transform(self, X):
        X['RSI_ema'] = self.rsi_ema(X['close'])
        X['RSI_ema'].fillna(method='bfill', inplace=True)  #fillna(X['RSI_ema'][self.period], inplace=True)
        
        X['rsi_threshold_ema'] = X['RSI_ema'].apply(
            lambda x: 'overbought_ema' if x >= self.h_bound 
            else 'oversold_ema' if x <= (100 - self.h_bound) 
            else 'above_50_less_70_ema' if (x >= 50 and x < self.h_bound)
            else 'below_50_above_30_ema'
        )
        
        #X = pd.get_dummies(X)  # Преобразуем категориальные переменные в дамми-переменные
        
        return pd.get_dummies(X)

class EMAAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, ema_short = 13, ema_long = 21, ema_longest = 134):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.ema_longest = ema_longest

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['EMA_short'] = talib.EMA(X.close, timeperiod=self.ema_short)
        X['EMA_short'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед
       
        X['EMA_long'] = talib.EMA(X.close, timeperiod=self.ema_long)
        X['EMA_long'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед
    
        X['EMA_longest'] = talib.EMA(X.close, timeperiod=self.ema_longest)
        X['EMA_longest'].fillna(method='bfill', inplace=True)  # Заполняем NaN ближайшими значениями вперед
    
        X['EMA_short_long'] = X.apply(lambda x: int(x['EMA_short'] >= x['EMA_long']), axis = 1)
        X['EMA_long_longest'] = X.apply(lambda x: int(x['EMA_long']>= x['EMA_longest']), axis = 1)
    
        X['close_vs_EMA_short'] = X.apply(lambda x: int(x['close'] >= x['EMA_short']), axis = 1)
        X['close_vs_EMA_long'] = X.apply(lambda x: int(x['close'] >= x['EMA_long']), axis = 1)
    
        return X

# RSI analyzer based on SMA
class RSIAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, period = 14, h_bound = 70):
        self.period = period
        self.h_bound = h_bound

    def fit(self, X, y=None):
        return self

    def transform(self, X): #calculate_rsi
        """
        Рассчитывает RSI и добавляет его в DataFrame.
        
        :param X: DataFrame с данными, содержащий колонку 'close'.
        :return: DataFrame с добавленным столбцом 'RSI' и 'rsi_threshold'.
        """
        if 'close' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")

        X['RSI'] = talib.RSI(X['close'], timeperiod=self.period)
        X['RSI'].fillna(method='bfill', inplace=True)  

        X['rsi_threshold'] = X['RSI'].apply(
            lambda x: 'overbought' if x >= self.h_bound 
            else 'oversold' if x <= (100 - self.h_bound) 
            else 'above_50_but_less_70' if (x >= 50 and x < self.h_bound) 
            else 'below_50_but_above_30'
        )
        
        # Преобразуем категориальные данные в дамми-переменные
        #X = pd.get_dummies(X, columns=['rsi_threshold'], drop_first=True)

        return pd.get_dummies(X, columns=['rsi_threshold'], drop_first=True)


class SMAAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, ma_short = 13, ma_long = 34, ma_longest = 233):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.ma_longest = ma_longest

    def fit(self, X, y=None):
        return self
    
    def MA(self, x_series, n):
        MA = x_series.rolling(window=n, min_periods=n).mean()
        #first_valid_value = MA[MA.first_valid_index()]  # Находим первое ненулевое значение
        MA.fillna(method='bfill', inplace=True)  #fillna(first_valid_value, inplace=True)
        return MA

    def transform(self, X):
        X['ma_short'] = self.MA(X['close'], self.ma_short)
        X['ma_long'] = self.MA(X['close'], self.ma_long)
        X['ma_longest'] = self.MA(X['close'], self.ma_longest)
        
        X['ma_short_long'] = X.apply(lambda x: int(x['ma_short'] >= x['ma_long']), axis=1)
        X['ma_long_longest'] = X.apply(lambda x: int(x['ma_long'] >= x['ma_longest']), axis=1)
    
        return X

class Debugger(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.head())  # Вывод первых нескольких строк DataFrame
        return X