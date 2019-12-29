import collections
import csv
import os
import types
from collections import OrderedDict

from G_save.abstract_save_handler import AbstractSaveHandler


class SaveHandler(AbstractSaveHandler):

    def __init__(self, config, experiment_results, filename):
        super().__init__(config)
        self._all_data = collections.OrderedDict(list(experiment_results.items()) + list(config.items()))
        self._filename = filename

    def save(self):
        self.__save_result()

    def __save_result(self):
        self.__create_if_not_exist()
        self.__curate_data()

        fieldnames = self.__set_headers()

        with open(self._filename, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(self._all_data)

    def __set_headers(self):
        field_names, old_rows = self.compute_headers()
        self.__rewrite_headers(field_names, old_rows)

        return field_names

    def compute_headers(self):
        if os.stat(self._filename).st_size == 0:
            return self._all_data.keys(), []

        with open(self._filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            field_names = next(csv_reader)

        with open(self._filename) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            old_rows = [row for row in csv_reader]

        new_field_names = list(self._all_data.keys())
        for key in field_names:
            if key not in new_field_names:
                new_field_names.append(key)

        return new_field_names, old_rows

    def __rewrite_headers(self, field_names, old_rows):
        with open(self._filename, 'w') as csv_file:
            w = csv.writer(csv_file)
            w.writerow(field_names)
            w = csv.DictWriter(csv_file, field_names)

            for row in old_rows:
                w.writerow(row)

    def __create_if_not_exist(self):
        if not os.path.exists(self._filename):
            with open(self._filename, 'w'): pass

    def __curate_data(self):
        confusion_matrix_str = "confusion_matrix"
        new_dict = OrderedDict()

        for key, value in self._all_data.items():
            if isinstance(value, float):
                new_dict[key] = '%.3f' % (value)
            elif key == confusion_matrix_str:
                new_dict[confusion_matrix_str + " (TP, FP, TN, FN)"] = value
            else:
                new_dict[key] = value

        self._all_data = new_dict

