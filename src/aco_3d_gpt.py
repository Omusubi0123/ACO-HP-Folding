import random

import numpy as np


class Ant:
    def __init__(self, sequence):
        self.sequence = sequence
        self.path = [(0, 0, 0)]  # 初期位置
        self.energy = 0

    def move(self, lattice, pheromones, alpha, beta):
        current_pos = self.path[-1]
        possible_moves = [
            (0, 1, 0),
            (1, 0, 0),
            (0, -1, 0),
            (-1, 0, 0),  # xy平面
            (0, 0, 1),
            (0, 0, -1),  # z軸方向
        ]
        next_moves = []

        for move in possible_moves:
            next_pos = (
                current_pos[0] + move[0],
                current_pos[1] + move[1],
                current_pos[2] + move[2],
            )
            if next_pos not in lattice:  # 自己交差を避ける
                h_neighbors = sum(
                    1
                    for neighbor in [
                        (next_pos[0] + dx, next_pos[1] + dy, next_pos[2] + dz)
                        for dx, dy, dz in possible_moves
                    ]
                    if neighbor in lattice and lattice[neighbor] == "H"
                )
                pheromone_level = pheromones.get((current_pos, next_pos), 1)
                heuristic_value = h_neighbors
                next_moves.append((next_pos, pheromone_level, heuristic_value))

        if not next_moves:
            return False  # 動けない場合

        # 確率に基づいて次の移動を選択
        probabilities = [(move[1] ** alpha) * (move[2] ** beta) for move in next_moves]
        probabilities = np.array(probabilities) / sum(probabilities)
        selected_move = random.choices(next_moves, probabilities)[0][0]

        self.path.append(selected_move)
        return True

    def calculate_energy(self, lattice):
        self.energy = 0
        for pos in self.path:
            if lattice[pos] == "H":
                neighbors = [
                    (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                    for dx, dy, dz in [
                        (0, 1, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (-1, 0, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ]
                ]
                self.energy += sum(
                    1
                    for neighbor in neighbors
                    if neighbor in lattice and lattice[neighbor] == "H"
                )
        self.energy //= 2  # 重複カウントを防ぐ


def run_aco(sequence, iterations, num_ants, alpha, beta, evaporation_rate):
    pheromones = {}
    best_energy = float("-inf")
    best_path = None

    for _ in range(iterations):
        ants = [Ant(sequence) for _ in range(num_ants)]
        lattice = {(0, 0, 0): sequence[0]}  # 格子初期化

        for ant in ants:
            for amino_acid in sequence[1:]:
                lattice[ant.path[-1]] = amino_acid
                if not ant.move(lattice, pheromones, alpha, beta):
                    break  # 動けない場合終了

            ant.calculate_energy(lattice)
            if ant.energy > best_energy:
                best_energy = ant.energy
                best_path = ant.path

        # フェロモンの更新
        for ant in ants:
            for i in range(len(ant.path) - 1):
                edge = (ant.path[i], ant.path[i + 1])
                pheromones[edge] = pheromones.get(edge, 0) + ant.energy

        # フェロモンの蒸発
        for edge in pheromones:
            pheromones[edge] *= 1 - evaporation_rate

    return best_energy, best_path


# 使用例
sequence = "HPHPPHHPHPPHPHHPPHPH"
best_energy, best_path = run_aco(
    sequence=sequence,
    iterations=100,
    num_ants=10,
    alpha=1,
    beta=2,
    evaporation_rate=0.1,
)

print(f"Best Energy: {best_energy}")
print(f"Best Path: {best_path}")
