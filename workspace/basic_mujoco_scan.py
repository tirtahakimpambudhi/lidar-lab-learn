import numpy as np
import mujoco
from mujoco_lidar import MjLidarWrapper, scan_gen

XML = '''
<mujoco>
  <worldbody>
    <geom name='lantai' type='plane' size='10 10 0.1'/>
    <body name='kotak' pos='2 0 0.5'>
      <geom name='kotak_geom' type='box' size='0.5 0.5 0.5'/>
    </body>
    <site name='lidar_site' pos='0 0 0.5'/>
  </worldbody>
</mujoco>
'''

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

lidar = MjLidarWrapper(model, site_name='lidar_site',backend='cpu', cutoff_dist=20.0)

theta, phi = scan_gen.create_lidar_single_line(horizontal_resolution=360)

r = lidar.trace_rays(data, theta, phi)
print('jumlah sinar      :', len(r))
print('sinar yang kena   :', int((r > 0).sum()))
print('sinar yang meleset:', int((r < 0).sum()), '(nilainya -1)')

i = int(np.argmin(np.abs(theta)))
print(f'jarak sinar ke depan: {r[i]:.3f} m  (harapan: 1.5 m, karena sisi depan kotak di x = 1.5)')

hit = r > 0
manual = np.stack([r*np.cos(phi)*np.cos(theta),
                   r*np.cos(phi)*np.sin(theta),
                   r*np.sin(phi)], axis=1)
pts = lidar.get_hit_points()
print('rumus sendiri == get_hit_points():', np.allclose(manual[hit], pts[hit], atol=1e-4))