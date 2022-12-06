
from environment import Anymal_runner
from models import DQLAgent
import numpy as np

import omni.appwindow  # Contains handle to keyboard
import sys
from utils import *


class DQLSim(Anymal_runner):
    def __init__(self, physics_dt, render_dt, reps, target):
        super().__init__(physics_dt, render_dt, reps=reps, target=target)

    def setup(self, episodes):
        
        self.max_steps = 500 # hyperparameter to control max number of phys steps
        self.current_step = 1
        
        self.episodes = episodes
        self.current_episode = 0
        self.done = False
        self.run_reward = 0

        self.agent = DQLAgent(state_space = 7, action_space = 7, gamma=.99, max_steps=self.max_steps)
        self._appwindow = omni.appwindow.get_default_app_window()
        
        state = self.get_state('/World/Anymal/base')
        self.state = np.reshape(state, [1, self.agent.osn])

        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step)
        print("_____________training setup________________")
        

    def on_physics_step(self, step_size) -> None:
        if self.current_episode == 0:
            self.reset_scene(initial_loc=[-8.6, 13.8, .49])
            self.current_episode += 1
        elif self.current_episode <= self.episodes:
            if self.current_step >= self.max_steps or self.done:
                # if done, or max steps reached, then reset the scene and variables

                self.reset_scene(initial_loc=[-8.6, 13.8, .49])
                self.current_episode += 1
                self.current_step = 1
                self.done = False

                print(f'Episode: {self.current_episode} | Steps: {self.max_steps} | Total reward: {self.run_reward} | Epsilon: {self.agent.epsilon}')
                self.agent.add_to_total_rewards(self.run_reward)
            
                self.run_reward = 0

                state = self.get_state('/World/Anymal/base')
                self.state = np.reshape(state, [1, self.agent.osn])
            
            self.memory.append(list(self.state[0]))
            
            action = self.get_action(self.state)
            action_omni = self.action_of_int(action)

            self._anymal.advance(step_size, action_omni)
            
            next_state = self.get_state('/World/Anymal/base')
            next_state = np.reshape(next_state, [1, self.agent.osn])

            # TODO: edit reward function
            reward = get_reward(next_state[0], self.target, self.current_step) # [0] required because next_state is reshaped for compatability with neural net
            self.done = not (is_standing(self.memory) and is_moving(tolerance = .01, mem_queue = self.memory)) or target_achieved(next_state[0], self.target, tolerance=.1)

            self.agent.memorize(self.state, action, reward, next_state, self.done, self.current_step)

            self.state = next_state
            self.run_reward = reward # keep track of the last run's reward, in case it terminates
            self.current_step += 1

            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.replay_batch()
        else:
            # Only save once
            self.save_model("qlearn", "iter_60")
            self.simulation_app.close()



def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """

    physics_dt = 1 / 100.0
    render_dt = 1 / 30.0

    runner = DQLSim(physics_dt=physics_dt, render_dt=render_dt, reps=40, target=[14, 2])
    runner.simulation_app.update()

    episodes = int(sys.argv[1])
    runner.setup(episodes = episodes)

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()

        

if __name__ == "__main__":
    main()
