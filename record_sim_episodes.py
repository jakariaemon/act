import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    1) Roll out a scripted policy in an ee_sim_env to collect a joint trajectory.
    2) Convert the gripper control info and incorporate it into the joint trajectory.
    3) Replay that joint trajectory in a sim_env to record all observations (including images & audio).
    4) Save this data to HDF5.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'angle'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Roll out EE-space scripted policy')
        # ------------------------------------------------------------
        # 1) ROLL OUT SCRIPTED POLICY in ee_sim_env
        # ------------------------------------------------------------
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()

        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        # Convert from gripper ctrl -> normalized gripper angles
        joint_traj = [ts.observation['qpos'] for ts in episode]
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6 + 7] = right_ctrl

        # We'll store the environment state (box pose, etc.) from step 0
        subtask_info = episode[0].observation['env_state'].copy()

        del env
        del episode
        del policy

        # ------------------------------------------------------------
        # 2) REPLAY in sim_env to record all final observations
        # ------------------------------------------------------------
        print('Replaying joint commands in sim_env')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info  # ensure same object configuration
        ts = env.reset()
        episode_replay = [ts]

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()

        for t in range(len(joint_traj)):  # note: one step longer than the number of actions
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        # ------------------------------------------------------------
        # 3) Prepare data dict for saving (including audio)
        # ------------------------------------------------------------
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # add image containers
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # ---------------------------
        # *** NEW: Add audio container
        # ---------------------------
        data_dict['/observations/audio'] = []

        # Because replay extends the episode length by 1, let's align the lengths
        joint_traj = joint_traj[:-1]        # drop last action
        episode_replay = episode_replay[:-1]  # drop last observation
        max_timesteps = len(joint_traj)

        # We now have 1:1 steps for joint_traj and episode_replay
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)

            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)

            # images
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(
                    ts.observation['images'][cam_name]
                )

            # *** NEW: audio
            if 'audio' in ts.observation:
                data_dict['/observations/audio'].append(ts.observation['audio'])
            else:
                # fallback if environment doesn't provide audio
                data_dict['/observations/audio'].append(
                    np.zeros((1, 1), dtype=np.float32)  # or skip
                )

        # ------------------------------------------------------------
        # 4) Save data to HDF5
        # ------------------------------------------------------------
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')

        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')

            # create dataset for images
            for cam_name in camera_names:
                image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype='uint8',
                    chunks=(1, 480, 640, 3)
                )

            # create qpos, qvel, action
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            # *** For audio, assume we know shape:
            # e.g. (max_timesteps, 40, 80). If shape can vary, we need a different approach.
            # Let's suppose shape is fixed for all timesteps:
            # We'll look at data_dict['/observations/audio'][0] to guess shape (40,80).
            if len(data_dict['/observations/audio']) > 0:
                audio_shape = data_dict['/observations/audio'][0].shape
                audio_ds = obs.create_dataset(
                    'audio',
                    (max_timesteps,) + audio_shape,
                    dtype='float32'
                )

            # Now actually write the arrays
            for name, array_list in data_dict.items():
                if name.startswith('/observations/images/'):
                    # e.g. /observations/images/angle
                    cam_name = name.split('/')[-1]
                    root['observations']['images'][cam_name][...] = array_list
                elif name == '/observations/qpos':
                    root['observations']['qpos'][...] = array_list
                elif name == '/observations/qvel':
                    root['observations']['qvel'][...] = array_list
                elif name == '/action':
                    root['action'][...] = array_list
                elif name == '/observations/audio':
                    # write to obs/audio dataset
                    if len(array_list) > 0:
                        root['observations']['audio'][...] = array_list
                else:
                    # If you had extra fields
                    pass

        print(f'Saving took {time.time() - t0:.1f} secs\n')

    # Summarize success rate
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False, default=1)
    parser.add_argument('--onscreen_render', action='store_true')
    args_parsed = vars(parser.parse_args())
    main(args_parsed)