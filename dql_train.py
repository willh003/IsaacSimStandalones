# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import numpy as np
import random
import sys
import os
import typing
from collections import deque
import math

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.quadruped.robots import Anymal
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, quat_to_euler_angles, euler_to_rot_matrix
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
from pxr import Gf, UsdGeom, Sdf
import omni.kit.commands
import omni.appwindow  # Contains handle to keyboard
import carb

from models import BasicAgent
from utils import *

class Anymal_runner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        """
        Summary

        creates the simulation world with preset physics_dt and render_dt and creates an anymal robot inside the warehouse

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        
        """
        
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        
        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # spawn warehouse scene
        prim = get_prim_at_path("/World/GroundPlane")
        if not prim.IsValid():
            prim = define_prim("/World/GroundPlane", "Xform")
            asset_path = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
            prim.GetReferences().AddReference(asset_path)

        self._anymal = self._world.scene.add(
            Anymal(
                prim_path="/World/Anymal",
                name="Anymal",
                usd_path=self.assets_root_path + "/Isaac/Robots/ANYbotics/anymal_c.usd",
                position=np.array([0, 0, 0]),
            )
        )
        self._world.reset()
        self._enter_toggled = 0
        self._base_command = np.zeros(3)
        self.usd_context = omni.usd.get_context()
        
        # specify the target [x, y] coordinates of the anymal in the scene
        self.target = [-13, 12] 

        # do nothing for the first few steps
        self.counter = 0

        # maintain a buffer of previous locations, to check if it has stopped moving
        self.memory = deque(maxlen=100)

        # map from the action space of the robot to commands to be performed by omniverse anymal
        self.omni_action_map = {"nothing": [0.0, 0.0, 0.0], "left": [0.0, 1.0, 0.0], "forward": [1.0, 0.0, 0.0], 
            "backward": [-1.0, 0.0, 0.0], "right": [0.0, -1.0, 0.0], "clock": [0.0, 0.0, -1.0], "counter-clock": [0.0, 0.0, 1.0]}

        # initialize a random first action
        self.action = random.randint(0, len(self.omni_action_map) - 1)

        # number of training repetitions to run
        self.reps = 40


    def on_physics_step_basic(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """
        # for the first 50 steps, do nothing, to give it time to get set
        # if self.counter < 50:
        #     self._anymal.advance(step_size, np.array([0.0, 0.0, 0.0]))
        #     self.counter += 1
        # else:
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

    def on_physics_step_train(self, step_size) -> None:
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
            self.save_model("iter_60")
            simulation_app.close()

    def on_physics_step_infer(self, step_size) -> None:

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
            

    def setup_basic(self) -> None:
        """
        [Summary]

        add physics callback
        
        """
        self.agent = BasicAgent()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step_basic)

    def setup_train(self, episodes):
        from models import DQLAgent
        
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

        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step_train)
        print("_____________training setup________________")
    
    def setup_infer(self, model_path, osn=7):
        from tensorflow.python.keras.models import load_model

        self.agent = load_model(model_path)
        self.osn = osn
        self.infer_counter = 0
        self._appwindow = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step_infer)


    def reset_scene(self, initial_loc=[0.0,0.0,0.0]):
        print("resetting")
        # maybe the robot prim itself stays in place, while its parts move after a reset
        # watch the anymal itself position before and after reset
        x = initial_loc[0]
        y = initial_loc[1]
        z = initial_loc[2]

        omni.kit.commands.execute('TransformPrimSRT',
            path=Sdf.Path('/World/Anymal/base'),
            new_translation=Gf.Vec3d(x, y, z),
            new_rotation_euler=Gf.Vec3d(0.0, -0.0, 0.0),
            new_rotation_order=Gf.Vec3i(0, 1, 2),
            new_scale=Gf.Vec3d(1.0, 1.0, 1.0),
            old_translation=Gf.Vec3d(0, 0, 0),
            old_rotation_euler=Gf.Vec3d(0.0, -0.0, 0.0),
            old_rotation_order=Gf.Vec3i(0, 1, 2),
            old_scale=Gf.Vec3d(1.0, 1.0, 1.0),
            )
        self.memory.clear()
        self.counter=0


    def get_action(self, state):

        # state has 3 components: loc, rot, target
        action = self.agent.get_action(state)
        return action


    def get_state(self, prim_path):
        stage = self.usd_context.get_stage()
        if not stage:
            return 

        # Get position directly from USD
        prim = stage.GetPrimAtPath(prim_path)

        loc = prim.GetAttribute("xformOp:translate") # VERY IMPORTANT: change to translate to make it translate instead of scale
        rot = prim.GetAttribute("xformOp:orient")
        rot = rot.Get()
        loc = loc.Get()
        str_nums = str(rot)[1:-1]
        str_nums = str_nums.replace(" ", "")
        str_nums = str_nums.split(',') 

        rot = []
        for s in str_nums:
            rot.append(float(s))

        # rot = euler_of_quat(rot)

        pose = list(loc)

        pose = [loc[0], loc[1], loc[2], rot[0], rot[1], rot[2], rot[3]]
        
        return pose

    def action_of_int(self, action_int):
        # the actions space of the robot
        actions = ["nothing", "left", "forward", "backward", "right", "clock", "counter-clock"]

        action_str = actions[action_int]
        return np.array(self.omni_action_map[action_str])

    def save_model(self, model_name):
        # rewards = self.rewards
        # plt.plot(range(1, len(rewards) + 1), rewards)
        # fig_path = os.path.join("models", "qlearn", "rewards", model_name)
        # plt.savefig(fig_path)
        print("reward")
        print(self.agent.tot_reward)
        model_path = os.path.join("models", "qlearn", model_name)

        # TODO: change self.agent name so it isn't dumb like this
        self.agent.save_model(model_path)

    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        while simulation_app.is_running():
            self._world.step(render=True)
        return


def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "basic"

    physics_dt = 1 / 100.0
    render_dt = 1 / 30.0

    runner = Anymal_runner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()

    if model == "basic":
        runner.setup_basic()

        # an extra reset is needed to register
        runner._world.reset()
        runner._world.reset()
        runner.run()
        simulation_app.close()
    elif model == "dql":
        episodes = int(sys.argv[2])
        runner.setup_train(episodes = episodes)

        # an extra reset is needed to register
        runner._world.reset()
        runner._world.reset()
        runner.run()
    elif model == "imitate":
        model_name = sys.argv[2]
        model_path = os.path.join("models", "imitation", model_name)
        runner.setup_infer(model_path, osn=7)
        # an extra reset is needed to register
        runner._world.reset()
        runner._world.reset()
        runner.run()
        
    else:
        print("NO MODEL SPECIFIED")
        


if __name__ == "__main__":
    main()
