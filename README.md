# 

# Sim2sim demos

## generate object xmls
python scripts/generate_g1_object_xmls.py

## locomotion
python test_loco_mw.py --offscreen --record --sim-hz 2000 --onnx src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_dr_0315.onnx --infer

## object trajectory tracking
python test_ott_mw.py --offscreen --record --sim-hz 2000 --policy ~/Desktop/codebase/unitreeG1/holosoma_wc/src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy_mlp_large.onnx --pkl src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl --clip-key GRAB_s1_cubemedium_pass_1 --lift --plot_joint_action
--stay  --gantry --init-timestep 130 --virtual --virtual_table

## Cabinet stocking

```
python test_cabinet_place_mw.py --obj-name spheremedium --stabilize-sec 1.0 --offscreen --record --release-with-hotdex --sim-hz 2000 --randomize --traj-speed-scale 0.8
```

## Repetitive pick-and-place
```
python test_repetitive_pnp.py --obj-name cubemedium --randomize --offscreen --record --release_with_hotdex --stabilize_sec 1 --sim-hz 2000
```

# Sim2sim for sim2real

## Locomotion with prior

In one terminal
```
source scripts/source_holomujoco_wc_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-43dof 

# 8 to lower the gantry / 7 to raise the gantry.
# press 9 to remove the gantry
```

```
source scripts/source_holoinference_mw.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-loco-prior     --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_dr_0306.onnx     --task.no-use-joystick     --task.interface lo

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-loco-prior     --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_dr_0315.onnx     --task.no-use-joystick     --task.interface lo --task.change_loco_order


python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-loco-prior     --task.model-path /home/rllab3/Desktop/codebase/unitreeG1/holosoma_wc/src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_mlp_0317.onnx     --task.no-use-joystick     --task.interface lo --task.change_loco_order --task.switch_hands


# press ] in the terminal
# press = to start locomotion
# WASD to move, QE to rotate
```

## Object tracking (bps)

```
sudo ip link set dev lo multicast on
source scripts/source_holomujoco_mw_setup.sh
python src/holosoma/holosoma/run_sim.py \
  robot:g1_43dof_cubemedium \
  --simulator.config.mujoco-motion-init.pkl-path "/home/rllab3/Desktop/codebase/unitreeG1/holosoma/src/holosoma/holosoma/data/motions/object_tracking/grab_omomo_selected_111_filtered.pkl" \
  --simulator.config.mujoco-motion-init.clip-key "GRAB_s1_cubemedium_pass_1" \
  --simulator.config.mujoco-motion-init.motion-start-timestep 0

python src/holosoma/holosoma/run_sim.py \
  robot:g1_43dof_apple \
  --simulator.config.mujoco-motion-init.pkl-path "/home/rllab3/Desktop/codebase/unitreeG1/holosoma/src/holosoma/holosoma/data/motions/object_tracking/grab_omomo_selected_111_filtered.pkl" \
  --simulator.config.mujoco-motion-init.clip-key "GRAB_s1_apple_pass_1" \
  --simulator.config.mujoco-motion-init.motion-start-timestep 0

```

Might need `pip install torch --index-url https://download.pytorch.org/whl/cpu`

```
source scripts/source_holoinference_mw.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps \
  --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy.onnx \
  --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl \
  --task.motion-clip-key GRAB_s1_apple_pass_1 \
  --task.no-use-joystick \
  --task.interface lo \
  --task.motion-start-timestep 0

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps \
  --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy_mlp_large_better.onnx \
  --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl \
  --task.motion-clip-key GRAB_s1_cubemedium_pass_1 \
  --task.no-use-joystick \
  --task.interface lo \
  --task.motion-start-timestep 20 \
  --task.use_gen_traj \
  --task.gen_traj_mode lift \
  --task.cache_world

# pcd based
source scripts/source_holoinference_mw.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object   --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/cube_tracking_policy.onnx   --task.no-use-joystick   --task.interface lo

enter : go to stiff
] : go to startup pose
d : stabilization
s : start
o : pause
oo : stop

```
`--task.use-gen-traj` for lift trajectory.

Use the same timestep value for both commands when you want MuJoCo init and policy playback to start from the same motion frame.

# Sim2Real 

## Hand control
```
cd ~/Desktop/codebase/unitreeG1/holosoma/unitree_sdk2_amazon/build/bin

./g1_dex3_example enp132s0 # ./g1_dex3_example for repeating both hands
```

## Locomotion with Prior
First Press L2+R2 and check the led turns to yellow
```
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-loco-prior \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_0315.onnx \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface enp132s0 \
    --task.change_loco_order \
    --task.log

# Debug hand-only mode: body holds current joints, Dex3 hands repeat open/grip forever
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-loco-prior \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_0315.onnx \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface enp132s0 \
    --task.change_loco_order \
    --task.log \
    --task.debug_hand \
    --task.debug_hand_action test_log/hand_joint_targets_latest.npz \
    --task.debug_hand_demo
```

# object tracking setup (FastLIO + apriltag + rviz)
```
./scripts/start_localization_session.sh
```

## debugging
```
python sim_replay.py --pkl logs/sim2real/wbt/wbt_log_enp132s0_20260320_232616.pkl --offscreen  --record --time 5
```


## Object trajectory following
```
# launch mujoco twin
python3 mujoco_twin.py --xml src/holosoma/holosoma/data/robots/g1/g1_object/g1_43dof_cubemedium.xml
```


```
holoin_wc
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps   --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy_mlp_large_better_local.onnx   --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl   --task.motion-clip-key GRAB_s1_cubemedium_pass_1   --task.use-joystick   --task.interface enp132s0   --task.motion-start-timestep 0 --task.use_gen_traj --task.gen_traj_mode lift --task.switch_hands --task.log --task.mujoco_twin 

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps   --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy_mlp_large_better.onnx   --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl   --task.motion-clip-key GRAB_s1_cubemedium_pass_1   --task.use-joystick   --task.interface enp132s0   --task.motion-start-timestep 0 --task.use_gen_traj --task.gen_traj_mode lift --task.switch_hands --task.log --task.mujoco_twin --task.cache_world
```
### hand velocity changed 
```
# right - use_gen_traj
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps   --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy_mlp_large_better.onnx   --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl   --task.motion-clip-key GRAB_s1_cubemedium_pass_1   --task.use-joystick   --task.interface enp132s0   --task.motion-start-timestep 0 --task.use_gen_traj --task.gen_traj_mode right --task.switch_hands --task.log --task.mujoco_twin --task.cache_world --task.fd_hand_vel

# record 

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps   --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/bps_policy_mlp_large_better.onnx   --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl   --task.motion-clip-key GRAB_s1_cubemedium_pass_1   --task.use-joystick   --task.interface enp132s0   --task.motion-start-timestep 0 --task.record_traj --task.switch_hands --task.log --task.mujoco_twin --task.cache_world --task.fd_hand_vel
```

### RSS rebuttal sim2real 
```
# use gen_traj : lift_back_left_down
# 1.2 times gain for hand motors 

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-object-bps   --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/object/0327.onnx   --task.motion-pkl-path /home/rllab3/Desktop/codebase/unitreeG1_mw/holosoma_wc/src/holosoma/holosoma/data/motions/motion_tracking/grab_omomo_selected_111_filtered.pkl   --task.motion-clip-key GRAB_s1_cubemedium_pass_1   --task.use-joystick   --task.interface enp132s0   --task.motion-start-timestep 0 --task.use_gen_traj --task.gen_traj_mode lift_back_left_down --task.switch_hands --task.log --task.mujoco_twin --task.fd_hand_vel --task.hand_gain_scale 1.2 

# joystick commands

A : interpolation (1)
X : stabilization (2)
start : starting motion (3)
B : when it fails, stabilization mode (4)
F1 : interpolate to default pose with stabilization policy (5)
X : go to starting pose (T pose) (6)
start : restart motion (7)

We can repeat (4)~(7)

```

## Use Fastlio 
```
ros_loc # env 
ros2 launch fast_lio mapping.launch.py fake_object:=true # terminal 1
ros2 launch livox_ros_driver2 msg_MID360_launch.py # terminal 2
ros2 run fast_lio holosoma_pose_bridge.py # terminal 3
```
### Robot dof pos 
```
holoin_wc
python read_dof_pos_real.py --ros

holoin_wc
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 ros2_joint_state_bridge.py

rviz2 -d ~/ws_loc/src/FAST_LIO_LOCALIZATION_HUMANOID/FAST_LIO/rviz/fastlio.rviz # rviz 
```


## Use Apriltag 
```
source ~/ros2_env.sh
ros2 launch realsense2_camera rs_launch.py
ros2 launch apriltag_ros realsense_apriltag.launch.py

python3 ~/ros2_ws/src/apriltag_ros/scripts/bundle_pose_node.py --ros-args -p bundle_config_file:=$HOME/ros2_ws/src/apriltag_ros/cfg/bundle.yaml -p camera_frame:=camera_link
```

## read rosbag 
```
 /usr/bin/python3 /home/rllab3/ws_loc/src/FAST_LIO_LOCALIZATION_HUMANOID/FAST_LIO/scripts/extract_false_detection_images.py   --rosbag_path /home/rllab3/Desktop/codebase/unitreeG1/holosoma_wc/logs/rosbag/rosbag_20260323_140500/rosbag_20260323_140500_0.mcap
```

## To use vscode debugger:
```
source scripts/source_holoinference_mw.sh
python3 -m debugpy --listen 5678 --wait-for-client path/to/your_script.py --arg1 --arg2
```

and set

```
{
    "configurations": [
        {
            "name": "Attach to sourced env",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "justMyCode": false
        }
    ]
}
```

# Holosoma

Holosoma (Greek: "whole-body") is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots, as well as motion retargeting. Supports locomotion (velocity tracking) and whole-body tracking tasks across multiple simulators (IsaacGym, IsaacSim, MJWarp, MuJoCo) with algorithms like PPO and FastSAC.

## Features

- **Multi-simulator support**: IsaacGym, IsaacSim, MuJoCo Warp (MJWarp), and MuJoCo (inference only)
- **Multiple RL algorithms**: PPO and FastSAC
- **Robot support**: Unitree G1 and Booster T1 humanoids
- **Task types**: Locomotion (velocity tracking) and whole-body tracking
- **Sim-to-sim and sim-to-real deployment**: Shared inference pipeline across simulation and real robot control
- **Motion retargeting**: Convert human motion capture data to robot motions while preserving interactions with objects and terrain
- **Wandb integration**: Video logging, automatic ONNX checkpoint uploads, and direct checkpoint loading from Wandb

## Repository Structure

```
src/
├── holosoma/              # Core training framework (locomotion & whole-body tracking)
├── holosoma_inference/    # Inference and deployment pipeline
└── holosoma_retargeting/  # Motion retargeting from human motion data to robots
```

## Documentation

- **[Training Guide](src/holosoma/README.md)** - Train locomotion and whole-body tracking policies in IsaacGym/IsaacSim
- **[Inference & Deployment Guide](src/holosoma_inference/README.md)** - Deploy policies to real robots or evaluate in MuJoCo simulation
- **[Retargeting Guide](src/holosoma_retargeting/README.md)** - Convert human motion capture data to robot motions

## Quick Start

### Setup

Choose the appropriate setup script based on your use case:

```bash
# For IsaacGym training
bash scripts/setup_isaacgym.sh

# For IsaacSim training
# Requires Ubuntu 22.04 or later due to IsaacSim dependencies
bash scripts/setup_isaacsim.sh

# For MJWarp training and MuJoCo simulation (inference)
bash scripts/setup_mujoco.sh

# For inference/deployment
bash scripts/setup_inference.sh

# For motion retargeting
bash scripts/setup_retargeting.sh
```

### Training

Train a G1 robot with FastSAC on IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

> **Note:** For headless servers, see the [training guide](src/holosoma/README.md#video-recording) for video recording configuration.

See the [Training Guide](src/holosoma/README.md) for more examples and configuration options.

### Quick Demo

We provide scripts to run the complete pipeline: (data downloading and processing for LAFAN), retargeting, data conversion, and whole-body tracking policy training.

```bash
# Run retargeting and whole-body tracking policy training using OMOMO data
bash demo_scripts/demo_omomo_wb_tracking.sh

# Run retargeting and whole-body tracking policy training using LAFAN data
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### Deployment & Evaluation

After training, deploy your policies:

- **Real Robot**: See [Real Robot Locomotion](src/holosoma_inference/docs/workflows/real-robot-locomotion.md) or [Real Robot WBT](src/holosoma_inference/docs/workflows/real-robot-wbt.md)
- **MuJoCo Simulation**: See [Sim-to-Sim Locomotion](src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md) or [Sim-to-Sim WBT](src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)

Or browse all deployment options in the [Inference & Deployment Guide](src/holosoma_inference/README.md).

### Demo Videos

Watch real-world deployments of Holosoma policies *(click thumbnails to play)*

<table>
  <tr>
    <th>G1 Locomotion</th>
    <th>T1 Locomotion</th>
    <th>G1 Dancing</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://youtu.be/YYMgj5BDIMI">
        <img src="https://img.youtube.com/vi/YYMgj5BDIMI/hqdefault.jpg" width="100%" alt="▶ G1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/Q6rNHJZ2a6Y">
        <img src="https://img.youtube.com/vi/Q6rNHJZ2a6Y/hqdefault.jpg" width="100%" alt="▶ T1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/ouPk69_eFfE">
        <img src="https://img.youtube.com/vi/ouPk69_eFfE/hqdefault.jpg" width="100%" alt="▶ G1 Dancing">
      </a>
    </td>
  </tr>
</table>


## Issue Reporting

We welcome feedback and issue reports to help improve holosoma. Please use issues to:

- Report bugs and technical issues
- Request new features

## Support

If you need help with anything aside from issues feel free to join our [discord server](https://discord.gg/TPupMvpqHc).

Use the discord to discuss larger plans and other more involved problems.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citation

If you use Holosoma in your research, please cite it according to the "Cite this repository" panel on the right sidebar of the Github repo.

## License

This project is licensed under the Apache-2.0 License.
