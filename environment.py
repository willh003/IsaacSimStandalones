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
import os
import typing
from collections import deque

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

from utils import *

class Anymal_runner(object):
    def __init__(self, physics_dt, render_dt, reps, target) -> None:
        """
        Summary

        creates the simulation world with preset physics_dt and render_dt and creates an anymal robot inside the warehouse

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        
        """
        self.simulation_app = simulation_app

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
        self.usd_context = omni.usd.get_context()
        
        # specify the target [x, y] coordinates of the anymal in the scene
        self.target = target

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
        self.reps = reps

        # subclass should override this
        self.agent = None

    def on_physics_step(self, step_size) -> None:
        """
        perform calculations, training, etc
        """
        
        pass
        
    
    def setup(self, **kwargs):
        """
        setup the model, and attach a physics callback to the world
        """
        pass


    def reset_scene(self, prim_path="/World/Anymal/base", initial_loc=[0.0,0.0,0.0]) :
        print("resetting")
        # maybe the robot prim itself stays in place, while its parts move after a reset
        # watch the anymal itself position before and after reset
        x = initial_loc[0]
        y = initial_loc[1]
        z = initial_loc[2]

        omni.kit.commands.execute('TransformPrimSRT',
            path=Sdf.Path(prim_path),
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
        """
        prim_path: path to a prim in the current stage
        returns: [translate, orient], where orient is the quaternion of the prim
        """
        stage = self.usd_context.get_stage()
        if not stage:
            return 

        # Get position directly from USD
        prim = stage.GetPrimAtPath(prim_path)

        loc = prim.GetAttribute("xformOp:translate")
        rot = prim.GetAttribute("xformOp:orient")
        rot = rot.Get()
        loc = loc.Get()
        str_nums = str(rot)[1:-1] # cheeky string conversion
        str_nums = str_nums.replace(" ", "")
        str_nums = str_nums.split(',') 

        rot = []
        for s in str_nums:
            rot.append(float(s))

        pose = list(loc)

        pose = [loc[0], loc[1], loc[2], rot[0], rot[1], rot[2], rot[3]]
        
        return pose

    def action_of_int(self, action_int):
        """
        action_int: an integer corresponding to an index in the actions array
        """

        actions = ["nothing", "left", "forward", "backward", "right", "clock", "counter-clock"]
        action_str = actions[action_int]

        return np.array(self.omni_action_map[action_str])

    def save_model(self, model_type, model_name):
        """
        model_type: string telling what folder to put model in (ex: "dql")
        model_name: string denoting what to name the model folder
        """
        model_path = os.path.join("models", model_type, model_name)

        self.agent.save_model(model_path)

    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        while self.simulation_app.is_running():
            self._world.step(render=True)
        return
