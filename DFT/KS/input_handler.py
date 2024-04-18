# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     input_handler
   Description :
   Author :       lizhenhui
   date：          2024/3/27
-------------------------------------------------
   Change Activity:
                   2024/3/27:
-------------------------------------------------
"""
# input_handler.py

class InputParameters:
    def __init__(self, atomic_coordinates, lattice_parameters, exchange_correlation_functional, precision):
        self.atomic_coordinates = atomic_coordinates
        self.lattice_parameters = lattice_parameters
        self.exchange_correlation_functional = exchange_correlation_functional
        self.precision = precision

def parse_input(input_file):
    # 解析输入文件，提取用户输入的信息
    with open(input_file, 'r') as f:
        lines = f.readlines()

    atomic_coordinates = []
    lattice_parameters = {}
    exchange_correlation_functional = None
    precision = None

    for line in lines:
        if line.strip():  # 跳过空行
            key, value = line.strip().split(':')
            key = key.strip().lower()

            if key == 'atomic_coordinates':
                atomic_coordinates.append([float(coord) for coord in value.split()])
            elif key == 'lattice_parameters':
                lattice_params = value.split()
                lattice_parameters['a'] = float(lattice_params[0])
                lattice_parameters['b'] = float(lattice_params[1])
                lattice_parameters['c'] = float(lattice_params[2])
            elif key == 'exchange_correlation_functional':
                exchange_correlation_functional = value.strip()
            elif key == 'precision':
                precision = float(value.strip())

    return InputParameters(atomic_coordinates, lattice_parameters, exchange_correlation_functional, precision)

# 示例用法
if __name__ == "__main__":
    input_file = "input.txt"  # 假设输入文件为 input.txt
    input_params = parse_input(input_file)

    print("Atomic Coordinates:", input_params.atomic_coordinates)
    print("Lattice Parameters:", input_params.lattice_parameters)
    print("Exchange-Correlation Functional:", input_params.exchange_correlation_functional)
    print("Precision:", input_params.precision)
