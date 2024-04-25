import copy
import math
import types


class Matrix:
    def __init__(self, matrix):
        self.__matrix = matrix
        self.__rows = len(matrix)
        self.__cols = len(matrix[0])
        self.T = self.__transpose()

    def __str__(self):
        return str(self.__matrix)

    def shape(self):
        return self.__rows, self.__cols

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Index must be a tuple of 2 integers")
            if key[0] >= self.__rows or key[1] >= self.__cols:
                raise IndexError("Index out of range")
            return self.__matrix[key[0]][key[1]]
        elif isinstance(key, int):
            if key >= self.__rows:
                raise IndexError("Index out of range")
            return self.__matrix[key]
        elif isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.__rows
            step = key.step or 1
            return Matrix([row[start:stop:step] for row in self.__matrix])
        else:
            raise TypeError("Invalid index type")

    def __transpose(self):
        transposed = [[] for _ in range(self.__cols)]
        for i in range(self.__rows):
            for j in range(self.__cols):
                transposed[j].append(self.__matrix[i][j])
        return transposed

    def dot(self, other):
        if self.__cols != other.__rows:
            raise ValueError(f"Incorrect Shape: Matrix 1: {self.shape()}"
                             f", Matrix 2: {other.shape()}")
        lst_c = [[0 for _ in range(other.__cols)] for _ in range(self.__rows)]
        for i in range(self.__rows):
            for j in range(other.__cols):
                for k in range(self.__cols):
                    lst_c[i][j] += self.__matrix[i][k] * other.__matrix[k][j]
        return Matrix(lst_c)

    def __add__(self, other):
        if isinstance(other, int):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] += other
            return Matrix(new_matrix)
        if isinstance(other, float):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] += other
            return Matrix(new_matrix)
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise IndexError(f"Invalid Shape: Matrix 1: {self.shape()}, Matrix 2: {other.shape()}")
            else:
                new_matrix = copy.deepcopy(self.__matrix)
                for i in range(self.__rows):
                    for j in range(self.__cols):
                        new_matrix[i][j] += other[i][j]
                return Matrix(new_matrix)

    def __sub__(self, other):
        if isinstance(other, int):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] -= other
            return Matrix(new_matrix)
        if isinstance(other, float):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] -= other
            return Matrix(new_matrix)
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise IndexError(f"Invalid Shape: Matrix 1:"
                                 f" {self.shape()}, Matrix 2: {other.shape()}")
            else:
                new_matrix = copy.deepcopy(self.__matrix)
                for i in range(self.__rows):
                    for j in range(self.__cols):
                        new_matrix[i][j] -= other[i][j]
                return Matrix(new_matrix)

    def __mul__(self, other):
        if isinstance(other, int):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] *= other
            return Matrix(new_matrix)
        if isinstance(other, float):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] *= other
            return Matrix(new_matrix)
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise IndexError(f"Invalid Shape: Matrix 1:"
                                 f" {self.shape()}, Matrix 2: {other.shape()}")
            else:
                new_matrix = copy.deepcopy(self.__matrix)
                for i in range(self.__rows):
                    for j in range(self.__cols):
                        new_matrix[i][j] *= other[i][j]
                return Matrix(new_matrix)

    def __truediv__(self, other):
        if isinstance(other, int):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] /= other
            return Matrix(new_matrix)
        if isinstance(other, float):
            new_matrix = copy.deepcopy(self.__matrix)
            for i in range(self.__rows):
                for j in range(self.__cols):
                    new_matrix[i][j] /= other
            return Matrix(new_matrix)
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise IndexError(f"Invalid Shape: Matrix 1: "
                                 f"{self.shape()}, Matrix 2: {other.shape()}")
            else:
                new_matrix = copy.deepcopy(self.__matrix)
                for i in range(self.__rows):
                    for j in range(self.__cols):
                        new_matrix[i][j] /= other[i][j]
                return Matrix(new_matrix)

    def __iter__(self):
        return iter(self.__matrix)

    def sum_elem(self):
        scalar = 0
        for i in range(self.__rows):
            for j in range(self.__cols):
                scalar += self.__matrix[i][j]
        return scalar

    def normalise(self, min_val=None, max_val=None):
        if isinstance(min_val, types.NoneType) is True and isinstance(max_val, types.NoneType) is True:
            min_val = [min(feature) for feature in self.T]
            max_val = [max(feature) for feature in self.T]
        normalised_data = []
        for data_point in self.__matrix:
            normalised_point = [(data_point[i] - min_val[i]) /
                                (max_val[i] - min_val[i]) for i in range
                                (len(data_point))]
            normalised_data.append(normalised_point)
        normalised_data = Matrix(normalised_data)
        return normalised_data, min_val, max_val

    def denormalise(self, min_val, max_val):
        denormalised_data = []
        for data_point in self.__matrix:
            denormalised_point = [data_point[i] * (max_val[i] - min_val[i]) + min_val[i]
                                  for i in range(len(data_point))]
            denormalised_data.append(denormalised_point)
        denormalised_data = Matrix(denormalised_data)
        return denormalised_data

    def split_into_batches(self, batch_size):
        num_batches = math.ceil(self.__rows / batch_size)
        batched_dataset = [[] for _ in range(num_batches)]
        for num_sample in range(self.__rows):
            batched_dataset[num_sample // batch_size].append(self[num_sample])
        batched_dataset = [Matrix(batch) for batch in batched_dataset]
        return batched_dataset

    def pow(self, degree):
        raised_matrix = Matrix([[elem ** degree for elem in row] for row in self.__matrix])
        return raised_matrix

    def get_column(self, index):
        if index < 0 or index >= self.__cols:
            raise IndexError("Column index out of range")
        column = [[row[index]] for row in self.__matrix]
        return Matrix(column)

    def set_column(self, index, new_column):
        if index < 0 or index >= self.__cols:
            raise IndexError("Column index out of range")
        if isinstance(new_column, list):
            if len(new_column) != self.__rows:
                raise ValueError("New column must be the same length as other columns")
        elif isinstance(new_column, Matrix):
            if new_column.__rows != self.__rows:
                raise ValueError("New column must be the same length as other columns")
        for i in range(self.__rows):
            self.__matrix[i][index] = new_column[i][0]

def matrix_sum(matrix_lst):
    sum_matrixes = Matrix([[0 for _ in range(matrix_lst[0].shape()[1])] for _ in range(matrix_lst[0].shape()[0])])
    for matrix in matrix_lst:
        sum_matrixes = sum_matrixes + matrix
    return sum_matrixes


