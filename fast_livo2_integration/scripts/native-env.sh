# Source this file from bash: source scripts/native-env.sh
fastlivo_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$fastlivo_repo/.native/env"
export ROS_HOME="$fastlivo_repo/.native/ros"
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
if [ -f "$fastlivo_repo/.native/ws/devel/setup.bash" ]; then
  source "$fastlivo_repo/.native/ws/devel/setup.bash"
fi
unset fastlivo_repo
