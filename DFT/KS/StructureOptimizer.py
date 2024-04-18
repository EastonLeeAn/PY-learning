# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     StructureOptimizer
   Description :
   Author :       lizhenhui
   date：          2024/3/27
-------------------------------------------------
   Change Activity:
                   2024/3/27:
-------------------------------------------------
"""
import numpy as np
from scipy.optimize import minimize

class StructureOptimizer:
    def __init__(self, initial_positions, potential_function):
        self.positions = initial_positions
        self.potential_function = potential_function

    def optimize(self):
        result = minimize(self.energy, self.positions.flatten(), method='CG', jac=self.gradient)
        optimized_positions = result.x.reshape((-1, 3))
        return optimized_positions

    def energy(self, positions):
        positions = positions.reshape((-1, 3))
        return self.potential_function(positions)

    def gradient(self, positions):
        positions = positions.reshape((-1, 3))
        gradient = np.zeros_like(positions)
        epsilon = 1e-6  # 梯度计算的微小偏移量

        for i in range(len(positions)):
            for j in range(3):
                positions[i, j] += epsilon
                energy_plus = self.potential_function(positions)

                positions[i, j] -= 2 * epsilon
                energy_minus = self.potential_function(positions)

                gradient[i, j] = (energy_plus - energy_minus) / (2 * epsilon)

                positions[i, j] += epsilon  # 恢复原始位置

        return gradient.flatten()

# 示例用法
if __name__ == "__main__":
    # 假设初始原子位置
    initial_positions = np.array([[0.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0],
                                   [2.0, 2.0, 2.0]])

    # 假设势能函数
    def potential_function(positions):
        # 这里假设势能是原子间距的平方和
        return np.sum(np.sum((positions[1:] - positions[:-1]) ** 2, axis=1))

    optimizer = StructureOptimizer(initial_positions, potential_function)
    optimized_positions = optimizer.optimize()

    print("Optimized Positions:")
    print(optimized_positions)
