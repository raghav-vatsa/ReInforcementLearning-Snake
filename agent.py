import torch
import random
import numpy as np
from collections import deque
from snake_gameAI import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE =1000
LR= 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # control the randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # popleft() once we exceed the memory
        self.model =Linear_QNet(11,256,3)# size of stae,hidden size(can be changed),output is 3
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)



    def get_state(self, game): # the 11 states
        head = game.snake[0] # get the position of the snake
        # 20 is the hardcoded value for block size. these values will tell us if the next step wil lead to a collision or not
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        # At 1 pt, only 1 of these values can be 1, rest will be 0
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight, so if we will collide in the next step of the direction we are going in or not
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #Danger right, so if we take a right from the current direction, will we collide or not
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)),

            #Danger left
            (dir_u and game.is_collision(point_l)) or 
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location, at a pt- 1 or 2 values will 1
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food down
            game.food.y > game.head.y # food up
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if max memory is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #;ist of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self,state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self,state):
        # random moves: trade off exploration / exploitation
        self.epsilon = 80 - self.n_game # so the more games we play, the smaller our epsilon will get
        final_move = [0,0,0]
        if random.randint(0,200)< self.epsilon: # the smaller epsilon will get, the less frequent we will make random moves
            move = random.randint(0,2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] =1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory also called experience and is very IMP for the agent, plot result
            game.reset()
            agent.n_game+=1
            agent.train_long_memory()

            if score> record:
                record = score
                agent.model.save()

            print('Game', agent.n_game, "Score", score, 'record: ', record)

            plot_scores.append(score)
            total_score+= score
            mean_score = total_score/agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()