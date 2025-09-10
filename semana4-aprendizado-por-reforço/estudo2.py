import gymnasium as gym
import numpy as np
import time
import random

def run(episodios=100, matrix=4, render=False, alpha=0.1, gamma=0.9, epsilon=0.2):
    env = gym.make(
        'FrozenLake-v1',
        map_name=f'{matrix}x{matrix}',
        is_slippery=False,
        render_mode='human' if render else None
    )

    q_table = np.zeros((matrix, matrix, 4))

    for ep in range(1, episodios + 1):
        estado = env.reset()[0]
        linha, coluna = np.unravel_index(estado, (matrix, matrix))

        done = False
        total_recompensa = 0
        passos = 0

        while not done:
            # Escolha da ação
            if random.uniform(0, 1) < epsilon:
                acao = env.action_space.sample()
            else:
                acao = np.argmax(q_table[linha, coluna])

            novo_estado, recompensa, done, truncado, _ = env.step(acao)

            novo_estado, recompensa, terminated, truncated, _ = env.step(acao)
            nova_linha, nova_coluna = np.unravel_index(novo_estado, (matrix, matrix))

            q_atual = q_table[linha, coluna, acao]
            q_max_futuro = np.max(q_table[nova_linha, nova_coluna])
            q_table[linha, coluna, acao] = \
                (1 - alpha) * q_atual + alpha * (recompensa + gamma * q_max_futuro)

            estado = novo_estado
            linha, coluna = nova_linha, nova_coluna
            total_recompensa += recompensa
            passos += 1

            done = terminated or truncated

            if render:
                time.sleep(0.5)

        print(f"Ep {ep:03} | Recompensa: {total_recompensa} | Passos: {passos}")

if __name__ == '__main__':
    run(render=True)