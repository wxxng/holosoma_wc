# 

# Sim2sim demos

## generate object xmls
python scripts/generate_g1_object_xmls.py

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
source scripts/source_holomujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-43dof 

# 8 to lower the gantry / 7 to raise the gantry.
# press 9 to remove the gantry
```

```
source scripts/source_holoinference_mw.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-43dof-loco-prior     --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_43dof/walk_prior_dr_0306.onnx     --task.no-use-joystick     --task.interface lo

# press ] in the terminal
# press = to start locomotion
# WASD to move, QE to rotate
```

## Object tracking (bps)

```
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
