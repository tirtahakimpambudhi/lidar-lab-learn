import numpy as np 
import mujoco 
import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt 
from mujoco_lidar import MjLidarWrapper, scan_gen 


model = mujoco.MjModel.from_xml_path('arena.xml')
data = mujoco.MjData(model)

robot_id = model.body('robot').id 
mocap_id = model.body_mocapid[robot_id]

def place_robot(x, y, yaw):
    