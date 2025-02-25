import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

class StockDataDownloader:
    def __init__(self, ticker = 'SBER', interval = 60):
        self.ticker = ticker
        self.interval = interval
        self.base_url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json'
    
    def download_prices(self, start_date, end_date):
        """Загрузка цен акций из API Московской биржи."""
        url = f"{self.base_url}?interval={self.interval}&from={start_date}&till={end_date}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            columns = data['candles']['columns']
            records = data['candles']['data']
            df = pd.DataFrame(records, columns=columns)
            df['begin'] = pd.to_datetime(df['begin'])
            df['end'] = pd.to_datetime(df['end'])
            return df
        else:
            print(f"Ошибка при получении данных: {response.status_code}")
            return pd.DataFrame()  # Возвращаем пустой DataFrame при ошибке
    
    def fetch_data(self, start_date, end_date):
        """Получение данных за указанный период."""
        end_date_datetime = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        # Начальная загрузка истории акции
        df = self.download_prices(start_date, end_date)
        df_new = df

        # Проверка наличия данных и выгрузка по 500 записей
        while not df.empty and df.end.max() < end_date_datetime and len(df_new)==500:
            df_new = self.download_prices(df.begin.max(), end_date)
            if not df_new.empty:
                df = pd.concat([df, df_new], axis=0)
            else:
                break  # Если новый DataFrame пустой, выходим из цикла
        
        # Удаляем дубликаты и лишние колонки
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if 'end' in df.columns:
            df.drop(['end', 'value'], axis=1, inplace=True)  # Удаляем лишние колонки
        
        return df

    def save_to_excel(self, df, filename):
        """Сохранение DataFrame в Excel файл."""
        df.to_excel(filename, index=False)

class PriceMovementAnalyzer:
    def __init__(self, delta = 0.05):
        self.delta = delta

    def growth_drop(self, df):
        """Создает колонку со значениями 1, если произошел рост на delta% или 0 при падении на delta%."""
        delta_max = 1 + self.delta
        delta_min = 1 - self.delta
        n_iter = df.shape[0]

        df['result'] = -1  # Определяем первоначально все значения как -1
        #df['n_close'] = 0  # Далее заменим значения на индекс часа, когда сделка закрылась

        # Расчет достигала ли стоимость инструмента стоимости равной стоимости покупки или продажи на каждом часе +/- delta%
        for i in tqdm(range(0, n_iter)):
            price_close = df.iloc[i].close
            price_max = round(price_close * delta_max, 2)
            price_min = round(price_close * delta_min, 2)
            j = i + 1
            
            while price_close > price_min and price_close < price_max and j < n_iter:
                if df.loc[j, 'low'] <= price_min:
                    price_close = df.loc[j, 'low']  # Достигли -delta% и необходимо выйти из цикла
                    df.loc[i, 'result'] = 0
                    df.loc[i, 'n_close'] = j # индекс строки когда минимальное значение достигло границы
                elif df.loc[j, 'high'] >= price_max:
                    price_close = df.loc[j, 'high']  # Достигли +delta% и необходимо выйти из цикла
                    df.loc[i, 'result'] = 1
                    df.loc[i, 'n_close'] = j # индекс строки когда минимальное значение достигло границы
                else:
                    j += 1
        
        return df

class TimeDifferenceCalculator:
    def __init__(self, data):
        self.data = data

    def calculate_diff_hours(self, i):
        """
        Рассчитывает разницу во времени между двумя строками по индексу 'n_close'.
        :param i: Индекс строки для которой рассчитывается разница.
        :return float: Разница в часах. Если индекс 'n_close' выходит за пределы данных, возвращает 0.
        """
        j = self.data.loc[i, 'n_close']
        
        if j >= len(self.data):
            return 0
        else:
            date1 = self.data.loc[j, 'begin']
            date2 = self.data.loc[i, 'begin']
            difference = date1 - date2
            return difference.total_seconds() / 3600

    def apply_to_dataframe(self):
        """
        Применяет функцию к каждой строке DataFrame и добавляет результат в новый столбец 'diff_hours'.
        :return pd.DataFrame: Обновленный DataFrame с новым столбцом.
        """
        self.data['diff_hours'] = 0.0
        for k in range(len(self.data)):
            self.data.loc[k, 'diff_hours'] = self.calculate_diff_hours(k)
        return self.data

