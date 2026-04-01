#!/usr/bin/env bash
# Start tmux session for localization + inference stack.
# Password for sudo: robot

set -u

SESSION="localization"
SUDO_PASS="robot"
FAKE_OBJECT=false
WINDOW_WIDTH=220
WINDOW_HEIGHT=50

for arg in "$@"; do
    case "$arg" in
        --fake_object)
            FAKE_OBJECT=true
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: $0 [--fake_object]" >&2
            exit 1
            ;;
    esac
done

# Helper: run command in clean env (no conda LD_LIBRARY_PATH pollution)
CLEAN="env -i HOME=$HOME USER=$USER PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin DISPLAY=${DISPLAY:-} XAUTHORITY=${XAUTHORITY:-}"
ROS2_WS="source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && cd ~/ros2_ws"
ROS_LOC="source /opt/ros/jazzy/setup.bash && source ~/ws_loc/install/setup.bash && cd ~/ws_loc/src"

commands=()

if [ "$FAKE_OBJECT" = true ]; then
    commands+=("$CLEAN bash -c '$ROS_LOC && ros2 launch fast_lio mapping.launch.py fake_object:=true'")
else
    commands+=("$CLEAN bash -c '$ROS_LOC && ros2 launch fast_lio mapping.launch.py'")
fi

commands+=("$CLEAN bash -c '$ROS_LOC && ros2 launch livox_ros_driver2 msg_MID360_launch.py'")
commands+=("$CLEAN bash -c '$ROS_LOC && rviz2 -d ~/ws_loc/src/FAST_LIO_LOCALIZATION_HUMANOID/FAST_LIO/rviz/fastlio.rviz'")

if [ "$FAKE_OBJECT" != true ]; then
    commands+=("$CLEAN bash -c '$ROS2_WS && ros2 launch realsense2_camera rs_launch.py config_file:=/home/rllab3/ros2_ws/rs_rgb_manual.yaml'")
    # commands+=("$CLEAN bash -c '$ROS2_WS && ros2 launch realsense2_camera rs_launch.py enable_depth:=false rgb_camera.enable_auto_exposure:=false exposure:=50 gain:=16 rgb_camera.color_profile:=640,480,30'")
    commands+=("$CLEAN bash -c '$ROS2_WS && ros2 launch apriltag_ros realsense_apriltag.launch.py'")
    commands+=("$CLEAN bash -c '$ROS2_WS && python3 ~/ros2_ws/src/apriltag_ros/scripts/bundle_pose_node.py --ros-args -p bundle_config_file:=\$HOME/ros2_ws/src/apriltag_ros/cfg/bundle.yaml'")
    commands+=("$CLEAN bash -c 'mkdir -p ~/Desktop/codebase/unitreeG1/holosoma_wc/rosbag && source /opt/ros/jazzy/setup.bash && ros2 bag record /camera/camera/color/image_raw /pelvis_pose_world /object_pose_world /object_pose_torso /bundle_pose /object_detected -o ~/Desktop/codebase/unitreeG1/holosoma_wc/rosbag/\$(date +%Y%m%d_%H%M%S)'")
fi

commands+=("echo ${SUDO_PASS} | sudo -S true && cd ~/Desktop/codebase/unitreeG1/holosoma_wc && source scripts/source_holoinference_wc_setup.sh && python read_dof_pos_real.py --ros")
commands+=("echo ${SUDO_PASS} | sudo -S true && cd ~/Desktop/codebase/unitreeG1/holosoma_wc && source scripts/source_holoinference_wc_setup.sh && source /opt/ros/jazzy/setup.bash && /usr/bin/python3 ros2_joint_state_bridge.py")
commands+=("$CLEAN bash -c '$ROS_LOC && ros2 run fast_lio holosoma_pose_bridge.py'")
commands+=("$CLEAN bash -c '$ROS_LOC && ros2 run fast_lio udp_traj_visualizer.py'")
commands+=("echo ${SUDO_PASS} | sudo -S true && cd ~/Desktop/codebase/unitreeG1/holosoma_wc && holomu_wc && python3 mujoco_twin.py --xml src/holosoma/holosoma/data/robots/g1/g1_object/g1_43dof_cubemedium.xml")

create_pane() {
    local session_target="$1"
    local largest_pane

    largest_pane="$(
        tmux list-panes -t "$session_target" -F '#{pane_id} #{pane_width} #{pane_height}' \
        | sort -k2,2nr -k3,3nr \
        | awk 'NR == 1 { print $1 }'
    )"

    tmux split-window -P -F '#{pane_id}' -t "$largest_pane"
}

# Kill existing session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create a detached session with a fixed window size so existing tmux clients do not shrink it.
tmux new-session -d -s "$SESSION" -x "$WINDOW_WIDTH" -y "$WINDOW_HEIGHT"
tmux set-window-option -t "$SESSION:0" window-size manual >/dev/null
tmux resize-window -t "$SESSION:0" -x "$WINDOW_WIDTH" -y "$WINDOW_HEIGHT" >/dev/null

pane_ids=()
pane_ids+=("$(tmux list-panes -t "$SESSION:0" -F '#{pane_id}' | head -n1)")

for ((i = 1; i < ${#commands[@]}; i++)); do
    if ! pane_id="$(create_pane "$SESSION:0" 2>/dev/null)"; then
        echo "Failed to create tmux pane $((i + 1)) of ${#commands[@]} in session '$SESSION'." >&2
        echo "tmux reported that there is no space for more panes. Increase the terminal size or reduce the command count." >&2
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        exit 1
    fi

    pane_ids+=("$pane_id")
    tmux select-layout -t "$SESSION:0" tiled >/dev/null
done

for ((i = 0; i < ${#commands[@]}; i++)); do
    tmux send-keys -t "${pane_ids[$i]}" "${commands[$i]}" Enter
done

tmux select-layout -t "$SESSION:0" tiled >/dev/null

# Attach to session.
if [ -n "${TMUX:-}" ]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach-session -t "$SESSION"
fi
