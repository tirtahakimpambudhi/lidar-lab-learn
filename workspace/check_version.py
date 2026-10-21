import mujoco 
import mujoco_lidar 
from mujoco_lidar import MjLidarWrapper, scan_gen 

print('Mujoco version:', mujoco.__version__)
print('Mujoco Lidar Version:', mujoco_lidar.__version__)
print('Function:', [n for n in dir(scan_gen) if n.startswith(('generate', 'scan'))])