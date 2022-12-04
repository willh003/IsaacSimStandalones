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
from omni.kit.commands import create
import omni.appwindow  # Contains handle to keyboard
import numpy as np
import carb
import csv
import sys
import os
# from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
import typing


class Anymal_runner(object):
    def __init__(self, physics_dt, render_dt, filename) -> None:
        """
        Summary

        creates the simulation world with preset physics_dt and render_dt and creates an anymal robot inside the warehouse

        Argument:
        physics_dt {float} -- Physics downtime of the scene.
        render_dt {float} -- Render downtime of the scene.
        
        """
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        
        assets_root_path = get_assets_root_path()

        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # spawn warehouse scene
        prim = get_prim_at_path("/World/GroundPlane")
        if not prim.IsValid():
            prim = define_prim("/World/GroundPlane", "Xform")
            asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
            prim.GetReferences().AddReference(asset_path)

        self._anymal = self._world.scene.add(
            Anymal(
                prim_path="/World/Anymal",
                name="Anymal",
                usd_path=assets_root_path + "/Isaac/Robots/ANYbotics/anymal_c.usd",
                position=np.array([0, 0, 0]),
            )
        )
        self.reset_scene()
        self._world.reset()
        self._enter_toggled = 0
        self._base_command = np.zeros(3)
        self.usd_context = omni.usd.get_context()
        


        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.0, 0.0, 0.0],
            "UP": [1.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }
        if filename != None:
            self.mode = "collect"
            self.collected_data = []
            self.data_path = os.path.join("data", filename + ".csv")
        else:
            self.mode = ""

    def setup(self) -> None:
        """
        [Summary]

        Set up keyboard listener and add physics callback
        
        """
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        """
        [Summary]

        Physics call back, switch robot mode and call robot advance function to compute and apply joint torque
        
        """
        if self.mode == "collect":
            state = self.get_state("/World/Anymal/base")
            action = list(self._base_command)
            self.collected_data.append(state + action)


        self._anymal.advance(step_size, self._base_command)

    def run(self) -> None:
        """
        [Summary]

        Step simulation based on rendering downtime
        
        """
        # change to sim running
        while simulation_app.is_running():
            self._world.step(render=True)
        return

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

    def save_data(self, path, data):
        with open(path, 'w') as f:
      
            # using csv.writer method from CSV package
            writer = csv.writer(f)
            writer.writerow(["dx", "dy", "dz", "qx", "qy", "qz", "qw", "right", "forward", "rotate"])
            
            writer.writerows(data)

    def reset_scene(self):
        print("resetting")

        # maybe the robot prim itself stays in place, while its parts move after a reset
        # watch the anymal itself position before and after reset


        omni.kit.commands.execute('TransformPrimSRT',
            path=Sdf.Path('/World/Anymal/base'),
            new_translation=Gf.Vec3d(-8.6, 13.8, .49),
            new_rotation_euler=Gf.Vec3d(0.0, -0.0, 0.0),
            new_rotation_order=Gf.Vec3i(0, 1, 2),
            new_scale=Gf.Vec3d(1.0, 1.0, 1.0),
            old_translation=Gf.Vec3d(0, 0, 0),
            old_rotation_euler=Gf.Vec3d(0.0, -0.0, 0.0),
            old_rotation_order=Gf.Vec3i(0, 1, 2),
            old_scale=Gf.Vec3d(1.0, 1.0, 1.0),
            )

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        [Summary]

        Keyboard subscriber callback to when kit is updated.
        
        """
        # reset event
        self._event_flag = False
        # when a key is pressed for released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] += np.array(self._input_keyboard_mapping[event.input.name])
            elif event.input.name == "S":
                self.save_data(self.data_path, self.collected_data)
            elif event.input.name == "R":
                self.reset_scene()
            elif event.input.name == "Q":
                self.collected_data = []


        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def get_pose(self, prim_path):
        stage = self.usd_context.get_stage()
        if not stage or self.current_path == "":
            return 

        # Get position directly from USD
        prim = stage.GetPrimAtPath(prim_path)

        loc = prim.GetAttribute("xformOp:translate") # VERY IMPORTANT: change to translate to make it translate instead of scale
        rot = prim.GetAttribute("xformOp:rotate")

        return loc, rot

    def follow_cam(self):
        # prim = get_prim_at_path("/World/Anymal/base")
        # matrix: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)


        # translate: Gf.Vec3d = matrix.ExtractTranslation()
        # rotation: Gf.Rotation = matrix.ExtractRotation()
        # print(type(rotation))
        # print(rotation)
        # rot_decompose = rotation.DecomposeRotation()
        # print('translate')
        # print(translate)
        # print('rot')
        # print(rot_decompose)

        pos =  self.get_position("/World/Anymal/base")


        # if s == "one":
        #     self.set_camera_view(np.array([1, 2, 3]), np.array([25, 50, 10]), '/World/GroundPlane/Camera')
        # elif s == "two":
        #     self.set_camera_view(np.array([25, 50, 10]), np.array([1, 2, 3]), camera_prim_path='/World/GroundPlane/Camera')
    
    def set_camera_view(self, eye, target, camera_prim_path = "/OmniverseKit_Persp"):
        """Set the location and target for a camera prim in the stage given its path

        Args:
            eye (typing.Optional[np.ndarray], optional): Location of camera. Defaults to None.
            target (typing.Optional[np.ndarray], optional): Location of camera target. Defaults to None.
            vel (float, optional): Velocity of the camera when controlling with keyboard. Defaults to 0.05.
            camera_prim_path (str, optional): Path to camera prim being set. Defaults to "/OmniverseKit_Persp".
        """
        meters_per_unit = get_stage_units()

        if eye is None:
            eye = np.array([1.5, 1.5, 1.5]) / meters_per_unit
        if target is None:
            target = np.array([0.01, 0.01, 0.01]) / meters_per_unit

        vel = .05

        vel = vel / meters_per_unit
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position(camera_prim_path, eye[0], eye[1], eye[2], True)
        viewport.set_camera_target(camera_prim_path, target[0], target[1], target[2], True)
        viewport.set_camera_move_velocity(vel)
        return



def main():
    """
    [Summary]

    Parse arguments and instantiate the ANYmal runner
    
    """
    physics_dt = 1 / 100.0
    render_dt = 1 / 30.0
    print(get_assets_root_path())

    if len(sys.argv) > 2 and sys.argv[1] == "collect":
        runner = Anymal_runner(physics_dt=physics_dt, render_dt=render_dt, filename=sys.argv[2])
    else:
        runner = Anymal_runner(physics_dt=physics_dt, render_dt=render_dt, filename=None)
    simulation_app.update()
    runner.setup()

    # an extra reset is needed to register
    runner._world.reset()
    runner._world.reset()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
