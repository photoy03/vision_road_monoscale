from setuptools import setup, Extension
from glob import glob
setup(ext_modules=[Extension('vision_road._native', ['vision_road/native.cpp'], include_dirs=['/usr/include/pybind11','/usr/include/opencv4'], libraries=['opencv_core','opencv_imgproc'], language='c++', extra_compile_args=['-O3','-std=c++17'], optional=True)], name='vision_road', version='0.1.0', packages=['vision_road'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/vision_road']),
                  ('share/vision_road', ['package.xml']),
                  ('share/vision_road/launch', glob('launch/*.launch.py')),
                  ('share/vision_road/config', glob('config/*.yaml')+glob('config/*.rviz')),
                  ('share/vision_road/config/calibration', glob('config/calibration/*'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='jaehun', maintainer_email='jaehun@example.com',
      description='Road curve detection and odometry aligned lane accumulation', license='Apache-2.0',
      entry_points={'console_scripts': ['road_node = vision_road.node:main', 'reference_pose = vision_road.reference_pose:main', 'kitti_replay = vision_road.kitti_replay:main']})
