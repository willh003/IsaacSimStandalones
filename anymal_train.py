# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.quadruped.robots import Anymal
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Gf, UsdGeom, Sdf
import omni.kit.commands
from omni.kit.commands import create
import omni.appwindow  # Contains handle to keyboard
import numpy as np
import carb
import random
from models import BasicModel
import sys
import os

# from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
import typing
from collections import deque
import math

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
            asset_path = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
            prim.GetReferences().AddReference(asset_path)

        self._anymal = self._world.scene.add(
            Anymal(
                prim_path="/World/Anymal",
                name="Anymal",
                usd_path=self.assets_root_path + "/Isaac/Robots/ANYbotics/anymal_c.usd",
                position=np.array([0, 0, .5]),
            )
        )
        self._world.reset()
        self._enter_toggled = 0
        self._base_command = np.zeros(3)
        self.usd_context = omni.usd.get_context()
        
        # specify the target [x, y] coordinates of the anymal in the scene
        self.target = [5, -4] 

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

        # literally just so I remember how to make a cube
        omni.kit.commands.execute('CreatePrimWithDefaultXform',
                    prim_type='Cube',
                    attributes={'size': 100, 'extent': [(50, 50, 50), (150, 150, 150)]})

    def reset_scene(self):
        print("resetting")

        # maybe the robot prim itself stays in place, while its parts move after a reset
        # watch the anymal itself position before and after reset


        omni.kit.commands.execute('TransformPrimSRT',
            path=Sdf.Path('/World/Anymal/base'),
            new_translation=Gf.Vec3d(0.0, 0.0, 0.0),
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
        

    def on_physics_step(self, step_size) -> None:
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

        if self.is_standing():
            #print("standing")
            self._anymal.advance(step_size, self.get_action(state))
        else:
            print("not standing")
            self._anymal.advance(step_size, np.array([0.0, 0.0, 0.0]))
            self.reset_scene()
            reset_state = self.get_state('/World/Anymal/base')
            print(reset_state[3:])

    def setup_basic(self) -> None:
        """
        [Summary]

        add physics callback
        
        """
        self.model = BasicModel()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step)


    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        while simulation_app.is_running():
            self._world.step(render=True)
        return

    def get_action(self, state):

        # state has 3 components: loc, rot, target
        action = self.model.get_action(state)
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

        # rot = self.euler_of_quat(rot)

        pose = list(loc)

        pose = [loc[0], loc[1], loc[2], rot[0], rot[1], rot[2], rot[3]]
        
        return pose

    def euler_of_quat(self, quats):
        x = quats[0]
        y = quats[1]
        z = quats[2]
        w = quats[3]
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1) * 180 / math.pi
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2) * 180 / math.pi
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4) * 180 / math.pi
     

        return roll_x, pitch_y, yaw_z # in degrees

    def rot_matrix_of_euler(self, xtheta, ytheta, ztheta):

        c1 = np.cos(xtheta * np.pi / 180)
        s1 = np.sin(xtheta * np.pi / 180)
        c2 = np.cos(ytheta * np.pi / 180)
        s2 = np.sin(ytheta * np.pi / 180)
        c3 = np.cos(ztheta * np.pi / 180)
        s3 = np.sin(ztheta * np.pi / 180)

        matrix=np.array([[c2*c3, -c2*s3, s2],
                    [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                    [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
        
        return matrix

    def quat_of_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return (qx, qy, qz, qw)

    def is_standing(self):

        mem = list(self.memory)

        if len(mem) < 20:
            return True

        for l in mem:
            qx, qy, qz, qw = l[3], l[4], l[5], l[6]

            # qx, qy, qz, qw = self.quat_of_euler(x, y, z)

            gz = qx*qx - qy*qy - qz*qz + qw*qw

            if gz > 0:
                return True

        return False

    def is_moving(self, tolerance):
        """
        True if the robot's x, y, or z position has changed by more than 
        [tolerance] in the last 20 physics steps
        """
        mem = list(self.memory)

        if len(mem) < 100:
            return True

        x0, y0, z0 = mem[0][0], mem[0][1], mem[0][2]
        for l in mem:
            x, y, z = l[0], l[1], l[2]

            if abs(x-x0) > tolerance or abs(y-y0) > tolerance or abs(z-z0) > tolerance:
                return True

        return False
    
    def get_euclidean_distance(self, state, target):
        print('target: ')
        print(target)
        print("state: ")
        print(state)
        return math.sqrt((target[0] - state[0]) ** 2 + (target[1] - state[1]) ** 2)

    def get_reward(self, state, target):
        return 1 / self.get_euclidean_distance(state, target)


    def target_achieved(self, state, target, tolerance):
        return self.get_euclidean_distance(state, target) < tolerance

    
    def setup_train(self, episodes):
        from models import DQLAgent
        
        self.max_steps = 500 # hyperparameter to control max number of phys steps
        self.current_step = 1
        
        self.episodes = episodes
        self.current_episode = 1
        self.done = False
        self.run_reward = 0

        self.model = DQLAgent(state_space = 7, action_space = 7, gamma=.99, max_steps=self.max_steps)
        self._appwindow = omni.appwindow.get_default_app_window()
        
        state = self.get_state('/World/Anymal/base')
        self.state = np.reshape(state, [1, self.model.osn])

        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step_train)

    
    def on_physics_step_train(self, step_size) -> None:
        if self.current_episode < self.episodes:
            if self.current_step >= self.max_steps or self.done:
                # if done, or max steps reached, then reset the scene and variables

                self.reset_scene()
                self.current_episode += 1
                self.current_step = 1
                self.done = False

                print(f'Episode: {self.current_episode} | Steps: {self.max_steps} | Total reward: {self.run_reward} | Epsilon: {self.model.epsilon}')
                self.model.add_to_total_rewards(self.run_reward)
            
                self.run_reward = 0

                state = self.get_state('/World/Anymal/base')
                self.state = np.reshape(state, [1, self.model.osn])
            
            self.memory.append(list(self.state[0]))
            
            action = self.get_action(self.state)
            action_omni = self.action_of_int(action)

            self._anymal.advance(step_size, action_omni)
            
            next_state = self.get_state('/World/Anymal/base')
            next_state = np.reshape(next_state, [1, self.model.osn])

            # TODO: edit reward function
            reward = self.get_reward(next_state[0], self.target) # [0] required because next_state is reshaped for compatability with neural net
            self.done = not self.is_standing() or self.target_achieved(next_state[0], self.target, tolerance=.1)

            self.model.memorize(self.state, action, reward, next_state, self.done, self.current_step)

            self.state = next_state
            self.run_reward += reward
            self.current_step += 1

            if len(self.model.memory) > self.model.batch_size:
                self.model.replay_batch()
        else:
            simulation_app.close()
    
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
        

        model_path = os.path.join("models", "qlearn", model_name)

        # TODO: change self.model name so it isn't dumb like this
        self.model.model.save(model_path)
        print(self.model.tot_reward)

def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "dql"

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
        runner.setup_train(episodes = 40)

        # an extra reset is needed to register
        runner._world.reset()
        runner._world.reset()
        runner.run()
        runner.save_model()
        
    else:
        print("NO MODEL SPECIFIED")
        


if __name__ == "__main__":
    main()
