import csv
import inspect
import os

from G_save.abstract_save_handler import AbstractSaveHandler

FILENAME = "experiments.csv"


class SaveHandler(AbstractSaveHandler):

    def __init__(self, config, experiment_results):
        super().__init__(config)
        self._all_data = dict(list(experiment_results.items()) + list(config.items()))

    def save(self):
        self.__save_result()

    def __save_result(self):
        self.__create_if_not_exist()
        self.__curate_data()

        fieldnames = self.__set_headers()

        with open(FILENAME, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(self._all_data)

    def __set_headers(self):
        field_names, old_rows = self.compute_headers()
        self.__rewrite_headers(field_names, old_rows)

        return field_names

    def compute_headers(self):
        with open(FILENAME) as csv_file:
            empty_file = os.stat(FILENAME).st_size == 0
            csv_reader = csv.reader(csv_file, delimiter=',')
            field_names = next(csv_reader) if not empty_file else []
            old_rows = [row for row in csv_reader] if not empty_file else []

        new_field_names = list(self._all_data.keys())
        for key in field_names:
            if key not in new_field_names:
                new_field_names.append(key)

        return new_field_names, old_rows

    def __rewrite_headers(self, field_names, old_rows):
        print(field_names)
        with open(FILENAME, 'w') as csv_file:
            w = csv.writer(csv_file)
            w.writerow(field_names)

            for row in old_rows:
                w.writerow(row)

    def __create_if_not_exist(self):
        if not os.path.exists(FILENAME):
            with open(FILENAME, 'w'): pass

    def __curate_data(self):
        confusion_matrix_str = "confusion_matrix"
        for key, value in self._all_data.items():
            if inspect.isclass(value):
                self._all_data[key] = value.__name__
            elif isinstance(value, float):
                self._all_data[key] = '%.3f' % (value)
            elif key == confusion_matrix_str:
                self._all_data[confusion_matrix_str + " (TP, FP, TN, FN)"] = value
                self._all_data.pop(confusion_matrix_str)

