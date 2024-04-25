from parallelized_functions.Matrix import Matrix, matrix_sum
import math
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import types


def compute_gradients(X, y, w, b, max_degree):
    N = y.shape()[0]
    y_pred = Matrix([[0] for _ in range(N)])
    for degree in range(max_degree):
        added = X.pow(degree + 1).dot(w.get_column(degree))
        y_pred = y_pred + added
    y_pred = y_pred + b
    error = y_pred - y
    w_grad = Matrix([[0] * w.shape()[1] for _ in range(X.shape()[1])])
    for degree in range(max_degree):
        w_grad.set_column(degree, Matrix(X.pow(degree + 1).T).dot(error) * (2 / N))
    b_grad = error.sum_elem() * (2 / N)
    return w_grad, b_grad


class ParMiniBatchGradientDescent:
    def __init__(self, X, y, learning_rate, num_iterations, batch_size, num_processes, stopping, max_degree):
        self.__batched_X = X.split_into_batches(batch_size)
        self.__num_processors = num_processes
        self.__degree = max_degree
        self.__X = X
        self.__stopping = stopping
        self.__y = y
        self.__batched_y = y.split_into_batches(batch_size)
        self.__batch_size = batch_size
        self.__num_batches = math.ceil(X.shape()[0] / batch_size)
        self.__epochs = num_iterations
        self.__learning_rate = learning_rate
        self.__w = Matrix([[0] * max_degree for _ in range(X.shape()[1])])
        self.__b = 0

    def optimize(self):
        num_iterations = self.__epochs
        progress_bar_width = 50
        print("Mini-Batch Gradient Descent Progress:")
        print("[" + " " * progress_bar_width + "]", end="\r")
        start = time.time()
        with ProcessPoolExecutor(max_workers=self.__num_processors) as executor:
            for epoch in range(self.__epochs):
                futures = [executor.submit(compute_gradients, batch_X, batch_y, self.__w, self.__b, self.__degree)
                           for batch_X, batch_y in zip(self.__batched_X, self.__batched_y)]

                gradients = [future.result() for future in futures]

                w_grad = matrix_sum([grad[0] for grad in gradients]) / len(gradients)
                b_grad = sum([grad[1] for grad in gradients]) / len(gradients)

                self.__w = self.__w - w_grad * self.__learning_rate
                self.__b = self.__b - b_grad * self.__learning_rate
                progress = (epoch + 1) / num_iterations
                filled_width = int(progress * progress_bar_width)
                progress_bar = "[" + "=" * filled_width + " " * (progress_bar_width - filled_width) + "]"
                print(f"{progress_bar} {progress:.0%}", end="\r")
                if abs(w_grad.sum_elem() / (w_grad.shape()[0] * w_grad.shape()[1])) < self.__stopping and abs(b_grad) < self.__stopping:
                    print(f"\r\nEarly Stopping on iteration {epoch}")
                    break
        print()
        end = time.time()
        print(f'Executed in {np.round(end - start, 3)} seconds')
        return end - start

    def predict(self, sample=None):
        if isinstance(sample, types.NoneType) is True:
            sample = self.__X
        y_pred = Matrix([[0] for _ in range(sample.shape()[0])])
        for degree in range(self.__w.shape()[1]):
            y_pred += sample.pow(degree + 1).dot(self.__w.get_column(degree))
        y_pred += self.__b
        return y_pred

    def get_coefficients(self):
        return self.__w, self.__b
