import unittest
from click.testing import CliRunner
from main import train, predict
import os
import csv


class TestTrain(unittest.TestCase):

    def setUp(self):
        # Настройка для каждого теста
        self.runner = CliRunner()
        self.data_path = '../data/singapore_airlines_reviews.csv'
        self.test_model_path = 'test_model.pkl'

    def tearDown(self):
        # Очистка после каждого теста
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)

    def test_train_with_test_data(self):
        # Тест обучения с тестовыми данными
        if os.path.exists(self.test_model_path):
            test_data_path = '../data/singapore_airlines_reviews.csv'
            result = self.runner.invoke(train, [
                '--data', self.data_path,
                '--test', test_data_path,
                '--model', self.test_model_path
            ])
            self.assertEqual(result.exit_code, 0)  # Успешное завершение
            self.assertIn("Accuracy on test set:", result.output)
            self.assertIn(f"Model saved to {self.test_model_path}", result.output)
        else:
            print("Path to model is incorrect")

    def test_train_with_split(self):
        # Тест работы --split
        if os.path.exists(self.test_model_path):
            result = self.runner.invoke(train, [
                '--data', self.data_path,
                '--split', 0.2,
                '--model', self.test_model_path
            ])
            self.assertEqual(result.exit_code, 0)
            self.assertIn(f"Model saved to {self.test_model_path}", result.output)
        else:
            print("Path to model is incorrect")

    def test_train_without_test_or_split(self):
        # Тест обучения без тестовых данных и без --split
        if os.path.exists(self.test_model_path):
            result = self.runner.invoke(train, [
                '--data', self.data_path,
                '--model', self.test_model_path
            ])
            self.assertEqual(result.exit_code, 0)
            self.assertIn(f"Model saved to {self.test_model_path}", result.output)
        else:
            print("Path to model is incorrect")

    def test_train_invalid_split(self):
        # Тест с некорректным аргументом --split
        if os.path.exists(self.test_model_path):
            result = self.runner.invoke(train, [
                '--data', self.data_path,
                '--split', 1.2,
                '--model', self.test_model_path
            ])
            self.assertEqual(result.exit_code, 2)  # Ошибка в аргументах
            self.assertIn("Split value must be between 0 and 1", result.output)
        else:
            print("Path to model is incorrect")

    def test_train_missing_data(self):
        # Тест с отсутствующим файлом с данными
        if os.path.exists(self.test_model_path):
            result = self.runner.invoke(train, [
                '--data', 'nonexistent_data.csv',
                '--model', self.test_model_path
            ])
            self.assertEqual(result.exit_code, 2)  # Ошибка в аргументах
            self.assertIn("Error: Invalid value for '--data': Path 'nonexistent_data.csv' does not exist.", result.output)
        else:
            print("Path to model is incorrect")


class TestPredict(unittest.TestCase):

    def setUp(self):
        # Настройка для каждого теста
        self.runner = CliRunner()
        self.data_path = '../data/singapore_airlines_reviews.csv'
        self.test_model_path = 'test_model.pkl'
        self.test_review = 'everything was perfect, thank you!'

    def tearDown(self):
        # Очистка после каждого теста
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)

    def test_predict_data_text(self):
        # Тест предсказания результата для текстовых данных
        if os.path.exists('test_model.pkl'):
            self.runner.invoke(train, ['--data', self.data_path, '--model', self.test_model_path, '--split', 0.2])
            result = self.runner.invoke(predict, ['--model', self.test_model_path, '--data', self.test_review])
            self.assertEqual(result.output, '1\n')
            self.assertEqual(result.exit_code, 0)
        else:
            print("Path to model is incorrect")

    def test_predict_data_csv(self):
        # Тест предсказания результата для данных из csv файла
        if os.path.exists('test_model.pkl'):
            reviews = [
                "everything was perfect, thanks",
                "wonderful experience and lovely crew",
                "terrible food",
                "i will not use this airline again"
            ]
            with open('../data/reviews.csv', 'wb') as file:
                writer = csv.DictWriter(file, fieldnames=['report'])
                writer.writeheader()
                for review in reviews:
                    writer.writerow({'report': review})

            self.runner.invoke(train, ['--data', self.data_path, '--model', self.test_model_path])
            result = self.runner.invoke(predict, ['--model', self.test_model_path, '--data', '../data/reviews.csv'])
            self.assertEqual(result.output, '1\n 1\n 0\n 0\n')
            self.assertEqual(result.exit_code, 0)
            os.remove('../data/reviews.csv')
        else:
            print("Path to model is incorrect")


if __name__ == '__main__':
    unittest.main()
