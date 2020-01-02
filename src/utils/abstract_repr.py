from abc import ABC, abstractmethod


class AbstractRepr:

    def __str__(self):
        return self.__compute_repr()

    def __repr__(self):
        return self.__compute_repr()

    def __compute_repr(self):
        repr = str(self.__class__.__name__) + "("

        for key, value in self._get_fields_for_repr().items():

            if repr[-1] != '(':
                repr += ", "

            if isinstance(value, float):
                value = '%.2f' % (value)
            repr += "{}({})".format(key, value)

        repr += ")"

        return repr

    @abstractmethod
    def _get_fields_for_repr(self):
        pass
