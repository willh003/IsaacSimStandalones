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
from pxr import Gf, UsdGeom
from omni.kit.commands import create
import omni.appwindow  # Contains handle to keyboard
import numpy as np
import carb
from models import BasicModel

# from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
import typing
from collections import deque

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
                position=np.array([0, 0, 0]),
            )
        )
        self._world.reset()
        self._enter_toggled = 0
        self._base_command = np.zeros(3)
        self.usd_context = omni.usd.get_context()
        
        # specify the target location of the anymal
        self.target = [-3, 4.6] 

        # specify the model to use
        self.model = BasicModel()

        # do nothing for the first few steps
        self.counter = 0

        # maintain a buffer of previous locations, to check if it has stopped moving
        self.memory = deque(maxlen=20)

        # literally just so I remember how to make a cube
        omni.kit.commands.execute('CreatePrimWithDefaultXform',
                    prim_type='Cube',
                    attributes={'size': 100, 'extent': [(50, 50, 50), (150, 150, 150)]})

    def reset_scene(self):
        self._world.scene.remove_object('/World/Anymal')
        self.memory = self.memory.clear()

        self._anymal = self._world.scene.add(
            Anymal(
                prim_path="/World/Anymal",
                name="Anymal",
                usd_path=self.assets_root_path + "/Isaac/Robots/ANYbotics/anymal_c.usd",
                position=np.array([0, 0, 0]),
            )
        )
        self.counter=0

    def on_physics_step(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """
        # for the first 50 steps, do nothing, to give it time to get set
        if self.counter < 50:
            self._anymal.advance(step_size, np.array([0.0, 0.0, 0.0]))
            self.counter += 1
        else:
            if self.is_moving(tolerance=1):
                self._anymal.advance(step_size, self.get_action())
            else: 
                self.reset_scene()

        #self._anymal.advance(step_size, self._base_command)

    def setup(self) -> None:
        """
        [Summary]

        Set up keyboard listener and add physics callback
        
        """
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

    def get_action(self):

        state = self.get_pose('/World/Anymal/base')

        # state has 3 components: loc, rot, target
        action = self.model.get_action(state)
        return action


    def get_pose(self, prim_path):
        stage = self.usd_context.get_stage()
        if not stage:
            return 

        # Get position directly from USD
        prim = stage.GetPrimAtPath(prim_path)

        loc = prim.GetAttribute("xformOp:translate") # VERY IMPORTANT: change to translate to make it translate instead of scale
        rot = prim.GetAttribute("xformOp:orient")
        rot = rot.Get()
        loc = loc.Get()
        print(rot)
        print(loc)
       # pose = list(loc)
        pose = [loc[0], loc[1], loc[2], rot[0], rot[1], rot[2], rot[3]]
        
        self.memory.append(pose)
        
        # pose and self.target are lists
        return {"pose": pose, "target": self.target}

    def is_moving(self, tolerance):
        """
        True if the robot's x, y, and z position has changed by more than 
        [tolerance] in the last 20 physics steps
        """
        mem = list(self.memory)
        if len(mem) < 20:
            return True

        x0, y0, z0 = mem[0][0], mem[0][1], mem[0][2]
        for l in mem:
            x, y, z = l[0], l[1], l[2]

            if abs(x-x0) > tolerance or abs(y-y0) > tolerance or abs(z-z0) > tolerance:
                return True
        return False


def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """
    physics_dt = 1 / 100.0
    render_dt = 1 / 30.0

    runner = Anymal_runner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()
    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
