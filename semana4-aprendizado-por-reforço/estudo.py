import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

def run(episodes=1, matrix=4, is_training=True, render=False):

    env = gym.make(
        'FrozenLake-v1', 
        map_name=f'{matrix}x{matrix}', 
        is_slippery=True, 
        render_mode='human' if render else None
    )
    
    q = np.empty((matrix, matrix), dtype=object)

    state = env.reset()[0]

    # 0  1  2  3
    # 4  5  6  7
    # 8  9 10 11
    # 12 13 14 15

    # Actions: 0=left 1=down 2=right 3=up

    while True:
        pos = np.unravel_index(state + 1, (matrix, matrix))
  
        if q[pos]:
            action = q[pos][0] if q[pos][2] else env.action_space.sample()
        else:
            while True:
                action = env.action_space.sample()
                
                if action == 0:
                    try: 
                        test_pos = np.unravel_index(state-1, (matrix, matrix)) 
                    except: test_pos = (0, 0)
                elif action == 1:
                    try: 
                        test_pos = np.unravel_index(state+matrix, (matrix, matrix)) 
                    except: test_pos = (0, 0)
                elif action == 2:
                    try: 
                        test_pos = np.unravel_index(state+1, (matrix, matrix)) 
                    except: test_pos = (0, 0)
                elif action == 3:
                    try: 
                        test_pos = np.unravel_index(state-matrix, (matrix, matrix)) 
                    except: test_pos = (0, 0)

                if test_pos == (0, 0):
                    continue
                else:
                    if q[test_pos] and q[test_pos][2]:
                        break
                    else:
                        break

        print('Ação:', action)
    
        new_state, reward, terminated, truncated, _ = env.step(action)

        row, col = np.unravel_index(new_state, q.shape)
        
        print('Estado atual:', state, 'Novo estado:', new_state)
        print('Reward:', reward)
        print('Terminou:', terminated)
 
        q[row, col] = [action, new_state, not terminated]
        print('Adicinou na tabela', action, new_state, not terminated)

        print('\n')

        state = new_state

        #time.sleep(2)
    
        if terminated or truncated:
            state = env.reset()[0]

        if reward == 1:
            print("Ganhouuu!")
            break

    env.close()

if __name__ == '__main__':
    run(render=True)