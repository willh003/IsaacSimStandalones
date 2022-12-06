
from environment import Anymal_runner
from models import BasicAgent
import numpy as np

import omni.appwindow  # Contains handle to keyboard
import sys
from utils import *


class BasicSim(Anymal_runner):
    def __init__(self, physics_dt, render_dt, reps, target):
        super().__init__(physics_dt, render_dt, reps=reps, target=target)

    def on_physics_step(self, step_size) -> None:
        """
        perform calculations, training, etc
        """
        
        state = self.get_state('/World/Anymal/base')
        self.memory.append(state)

        if is_standing(self.memory):
            #print("standing")
            self._anymal.advance(step_size, self.get_action(state))
        else:
            print("not standing")
            self._anymal.advance(step_size, np.array([0.0, 0.0, 0.0]))
            self.reset_scene()
            reset_state = self.get_state('/World/Anymal/base')
            print(reset_state[3:])

    def setup(self) -> None:
        """
        [Summary]

        add physics callback
        
        """
        self.agent = BasicAgent()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step)


def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """

    physics_dt = 1 / 100.0
    render_dt = 1 / 30.0

    runner = BasicSim(physics_dt=physics_dt, render_dt=render_dt, reps=40, target=[14, 2])
    runner.simulation_app.update()

    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    runner.simulation_app.close()

        

if __name__ == "__main__":
    main()
