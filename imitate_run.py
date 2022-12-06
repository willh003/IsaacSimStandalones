  
  
from environment import Anymal_runner
import numpy as np
import sys
import os
from tensorflow.python.keras.models import load_model
import omni.appwindow  # Contains handle to keyboard
import sys
from utils import *


class ImitateSim(Anymal_runner):
    def __init__(self, physics_dt, render_dt, reps, target):
        super().__init__(physics_dt, render_dt, reps=reps, target=target)

    def setup(self, model_path, osn=7):
        self.agent = load_model(model_path)
        self.osn = osn
        self.infer_counter = 0
        self._appwindow = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step)
    
    def on_physics_step(self, step_size) -> None:

        state = self.get_state('/World/Anymal/base')
        self.memory.append(state)

        if self.infer_counter == 0:
            self.reset_scene(initial_loc=[-8.6, 13.8, .49])
            self.infer_counter += 1
        elif is_standing(self.memory) and self.infer_counter >= 100:
            #print("standing")
            state = np.reshape(state, [1, self.osn])
            action = self.agent.predict(state, verbose=0)[0]
            print(action)
            self._anymal.advance(step_size, action)
        else:
            self._anymal.advance(step_size, np.array([0.0, 0.0, 0.0]))
            self.reset_scene(initial_loc=[-8.6, 13.8, .49])
            if self.infer_counter < 100:
                self.infer_counter += 1
            


def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """

    physics_dt = 1 / 100.0
    render_dt = 1 / 30.0

    runner = ImitateSim(physics_dt=physics_dt, render_dt=render_dt, reps=40, target=[14, 2])
    runner.simulation_app.update()

    model_name = sys.argv[1]
    model_path = os.path.join("models", "imitation", model_name)
    runner.setup(model_path=model_path)

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    runner.simulation_app.close()

        

if __name__ == "__main__":
    main()
