# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     expand_POSCAR
   Description :
   Author :       lizhenhui
   date：          2024/4/15
-------------------------------------------------
   Change Activity:
                   2024/4/15:
-------------------------------------------------
"""
def expand_POSCAR(input_file, output_file, expansion=(2, 2, 2)):
    with open(input_file, 'r') as f:
        # 读取POSCAR文件内容
        poscar = f.readlines()

    # 提取POSCAR文件中的晶胞信息
    lattice = [list(map(float, line.split())) for line in poscar[2:5]]
    atoms = poscar[5].split()
    atom_counts = list(map(int, poscar[6].split()))

    # 计算新的晶胞大小
    expanded_lattice = [[lattice[i][j] * expansion[j] for j in range(3)] for i in range(3)]
    expanded_atom_counts = [count * expansion[0] * expansion[1] * expansion[2] for count in atom_counts]

    # 生成新的原子位置
    atom_positions = []
    for line in poscar[8:]:
        atom_positions.append([float(coord) for coord in line.split()[:3]])

    expanded_atom_positions = []
    for i in range(expansion[0]):
        for j in range(expansion[1]):
            for k in range(expansion[2]):
                for pos in atom_positions:
                    expanded_atom_positions.append([pos[0] + i, pos[1] + j, pos[2] + k])

    # 写入新的POSCAR文件
    with open(output_file, 'w') as f:
        f.write("".join(poscar[:2]))
        f.write("\n".join(" ".join(map(str, vec)) for vec in expanded_lattice))
        f.write("\n")
        f.write(" ".join(atoms))
        f.write("\n")
        f.write(" ".join(map(str, expanded_atom_counts)))
        f.write("\n")
        f.write("Direct\n")
        f.write("\n".join(" ".join(map(str, pos)) for pos in expanded_atom_positions))

# 示例用法
expand_POSCAR("input.POSCAR", "output.POSCAR", expansion=(2, 2, 2))
