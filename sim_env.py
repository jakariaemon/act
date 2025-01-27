import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

# Global variable that external code can modify
BOX_POSE = [None]

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control.
    Action space: [left_arm_qpos(6), left_gripper_positions(1), right_arm_qpos(6), right_gripper_positions(1)]
    Observation space:
      {
        "qpos": (14,),
        "qvel": (14,),
        "env_state": ...,
        "images": {"main": (480,640,3) ...},
        "audio": (for example, (40,80))  <-- newly added dummy audio
      }
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(
            physics, task,
            time_limit=20,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False
        )
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(
            physics, task,
            time_limit=20,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False
        )
    else:
        raise NotImplementedError

    return env


class BimanualViperXTask(base.Task):
    """
    Base class for bimanual manipulation tasks. 
    We define how actions are interpreted and how observations are returned.
    """

    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        """
        Converts the simplified action space into joint positions & gripper positions for Mujoco.
        """
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        # Each gripper has 2 joints that move symmetrically
        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([
            left_arm_action,
            full_left_gripper_action,
            right_arm_action,
            full_right_gripper_action
        ])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # split left side, right side
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]

        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]

        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        # split left side, right side
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]

        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]

        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        """
        Override in child classes to return object states, etc.
        """
        raise NotImplementedError

    def get_observation(self, physics):
        """
        Returns a dictionary of all relevant robot state and camera images.
        Also includes a dummy 'audio' entry with random data.
        """
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)

        # Render camera images
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        ################################################################
        #  DUMMY AUDIO: create random data, shape (40,80) float32
        ################################################################
        audio_shape = (40, 80)
        audio_dummy = np.random.randn(*audio_shape).astype(np.float32)
        obs['audio'] = audio_dummy

        return obs

    def get_reward(self, physics):
        # to be implemented in child classes
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    """
    A specialized environment for transferring a cube from one gripper to the other.
    """
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        # Use BOX_POSE[0] as the box location
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # return the last 7 elements of qpos, which presumably represent the box
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # check contacts
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    """
    A specialized environment for inserting a peg into a socket.
    """
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        # Use BOX_POSE[0] as the peg & socket positions
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-14:] = BOX_POSE[0]  # Two objects, each 7 dof presumably
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # last 14 elements: peg + socket states
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or
            ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or
            ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or
            ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs or
            ("socket-2", "table") in all_contact_pairs or
            ("socket-3", "table") in all_contact_pairs or
            ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs or
            ("red_peg", "socket-2") in all_contact_pairs or
            ("red_peg", "socket-3") in all_contact_pairs or
            ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # both are grasped
            reward = 1
        if (touch_left_gripper and touch_right_gripper and
                not peg_touch_table and not socket_touch_table):
            # both objects are lifted
            reward = 2
        if peg_touch_socket and not peg_touch_table and not socket_touch_table:
            # peg near/in socket
            reward = 3
        if pin_touched:
            # successful insertion
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    """
    Helper for teleoperation. 
    Takes the real robot states from master_bot_left/right, and transforms 
    them into the 14D action for the sim environment.
    """
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]

    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """
    Testing teleoperation in sim with ALOHA hardware. 
    Requires hardware and ALOHA repo to work. 
    """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    # Example box pose
    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # Source of data
    master_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s", group_name="arm", gripper_name="gripper",
        robot_name='master_left', init_node=True
    )
    master_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s", group_name="arm", gripper_name="gripper",
        robot_name='master_right', init_node=False
    )

    # Setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]

    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)

if __name__ == '__main__':
    test_sim_teleop()