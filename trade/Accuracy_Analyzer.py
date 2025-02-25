import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import itertools
import random
import math

class UniqueParameterCombinator:
    def __init__(self, params_dict, percent_to_process=70):
        # Генерируем все возможные комбинации
        self._all_combinations = list(itertools.product(*params_dict.values()))
        
        # Словарь для маппинга ключей
        self._param_keys = list(params_dict.keys())
        
        # Перемешиваем комбинации
        random.shuffle(self._all_combinations)
        
        # Трекер использованных комбинаций
        self._used_combinations = set()
        
        # Расчет количества комбинаций для обработки
        total_combinations = len(self._all_combinations)
        self._combinations_to_process = math.ceil(total_combinations * (percent_to_process / 100))
        
        # Счетчик обработанных комбинаций
        self._processed_count = 0

    def get_next_unique_combination(self):
        # Проверка лимита обработанных комбинаций
        if self._processed_count >= self._combinations_to_process:
            return None

        for combination in self._all_combinations:
            # Преобразуем комбинацию в хешируемый формат
            hashable_combination = hash(combination)
            
            # Проверяем, не использовалась ли комбинация ранее
            if hashable_combination not in self._used_combinations:
                # Отмечаем как использованную
                self._used_combinations.add(hashable_combination)
                
                # Создаем словарь параметров
                current_params = dict(zip(self._param_keys, combination))
                
                # Увеличиваем счетчик обработанных комбинаций
                self._processed_count += 1
                
                return current_params
        
        # Если все комбинации исчерпаны
        return None

class ModelEvaluator(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        """
        Инициализация класса ModelEvaluator.

        :param model: Модель машинного обучения.
        """
        self.model = model

    def fit(self, X_train_scaled, y_train):
        """
        Обучает модель на тренировочных данных.

        :param X_train_scaled: Масштабированные тренировочные данные.
        :param y_train: Целевые значения для тренировочных данных.
        """
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test_scaled):
        """
        Выполняет прогнозирование на тестовых данных.

        :param X_test_scaled: Масштабированные тестовые данные.
        :return: Прогнозируемые значения.
        """
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test_scaled, y_test):
        """
        Оценивает модель и возвращает метрики производительности.

        :param X_test_scaled: Масштабированные тестовые данные.
        :param y_test: Целевые значения для тестовых данных.
        :return: DataFrame с результатами оценки модели и TPR/FPR для разных порогов.
        """
        # Прогнозирование вероятностей
        method_probs = self.model.predict_proba(X_test_scaled)[:, 1]

        # Визуализация ROC-кривой
        self.plot_roc_curve(y_test, method_probs)

        # Расчет метрик
        y_pred = self.predict(X_test_scaled)
        
        precision = [precision_score(y_test, y_pred, pos_label=x) for x in [0, 1]]
        recall = [recall_score(y_test, y_pred, pos_label=x) for x in [0, 1]]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = [f1_score(y_test, y_pred, pos_label=x) for x in [0, 1]]
        
        roc_auc = [roc_auc_score(y_test, self.model.predict_proba(X_test_scaled)[:,x]) for x in [0,1]]

        # Создание DataFrame с результатами
        mod_results = pd.DataFrame({
            'model': self.model.__class__.__name__,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy
        })

        mod_results.reset_index(names='class', inplace=True)

        # Расчет TPR и FPR при различных порогах
        thresholds = np.arange(0.0, 1.1, 0.1)
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            y_pred_thresholded = (method_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresholded).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_fpr_df = pd.DataFrame({
            'threshold': thresholds,
            'TPR': tpr_list,
            'FPR': fpr_list
        })

        return mod_results, tpr_fpr_df

    def plot_roc_curve(self, y_test, method_probs):
        fpr, tpr, thresholds = roc_curve(y_test, method_probs)
        roc_auc = auc(fpr, tpr)
    
        plt.figure(figsize = (10,3))
        plt.plot(fpr, tpr, color = 'blue', label = 'ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0,1], [0,1], color = 'red', linestyle = '--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROCharacteristics for {self.model}')
        plt.legend(loc = "lower right")
        plt.show()


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column=None, apply_scaling=True, apply_balancing=False):
        """
        Инициализация класса DataPreprocessor.

        :param target_column: Имя колонки с целевыми значениями (по умолчанию None).
        :param apply_scaling: Флаг для применения нормализации (по умолчанию True).
        :param apply_balancing: Флаг для применения SMOTE (по умолчанию True).
        """
        self.target_column = target_column
        self.apply_scaling = apply_scaling
        self.apply_balancing = apply_balancing
        self.scaler = MinMaxScaler() if apply_scaling else None
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Обучает предобработчик на данных. В данном случае метод не требует выполнения операций.

        :param X: Входной DataFrame с признаками.
        :param y: Не используется, но требуется для совместимости с интерфейсом.
        :return: self
        """
        # Метод fit оставляем пустым, так как все операции будут выполнены в transform
        self.is_fitted = True  # Устанавливаем флаг после "обучения"
        return self

    def transform(self, X):
        """
        Применяет нормализацию и балансировку к данным.

        :param X: Входной DataFrame с признаками и целевой переменной.
        :return: Нормализованный DataFrame с признаками (и сбалансированный набор целевых значений, если применимо).
        """
        if not self.is_fitted:
            raise RuntimeError("You must fit the preprocessor before transforming.")

        # Проверка наличия целевой переменной
        if self.target_column and self.target_column in X.columns:
            # Извлечение целевой переменной
            y = X[self.target_column]
            X_features = X.drop(columns=[self.target_column])
            
            # Нормализация признаков
            if self.apply_scaling:
                X_scaled = self.scaler.fit_transform(X_features)  # Применяем fit_transform для нормализации
                X_processed = pd.DataFrame(X_scaled, columns=X_features.columns)
            else:
                X_processed = X_features

            # Балансировка тренировочного набора с помощью SMOTE (если y предоставлен)
            if self.apply_balancing:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_processed, y)
                return pd.DataFrame(X_resampled, columns=X_processed.columns), pd.Series(y_resampled)

            return X_processed, y  # Возвращаем нормализованные данные и целевые значения
        
        else:
            # Если целевая колонка отсутствует, просто нормализуем данные
            if self.apply_scaling:
                X_scaled = self.scaler.fit_transform(X)  # Применяем fit_transform для нормализации
                return pd.DataFrame(X_scaled, columns=X.columns)
            else:
                return X  # Возвращаем исходные данные без изменений




