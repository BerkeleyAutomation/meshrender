"""
Random variables for sampling camera poses.
Author: Jeff Mahler
"""
import copy
import logging

import numpy as np
import scipy.stats as ss

from autolab_core import Point, RigidTransform, RandomVariable, transformations
from autolab_core.utils import sph2cart, cart2sph
from perception import CameraIntrinsics, RenderMode

from .camera import VirtualCamera

class CameraSample(object):
    """Struct to encapsulate the results of sampling a camera and its pose.

    Attributes
    ----------
    object_to_camera_pose : autolab_core.RigidTransform
        A transfrom from the object frame to the camera frame.
    camera_intr : perception.CameraIntrinsics
        The camera's intrinsics.
    radius : float
        The distance from the center of the object's frame to the camera's eye.
    elev : float
        The angle of elevation to the camera from the object frame.
    az : float
        The angle of rotation of the camera's eye about the object's z axis, starting
        from the x axis.
    roll : float
        The roll angle of the camera about its viewing axis.
    tx : float
        The x-axis translation of the object.
    ty : float
        The y-axis translation of the object.
    focal : float
        The focal length of the camera.
    cx : float
        The x-axis optical center of the camera.
    cy : float
        The y-axis optical center of the camera.
    """
    def __init__(self, camera_to_world_pose, camera_intr,
                 radius, elev, az, roll, tx=0, ty=0, focal=0, 
                 cx=0, cy=0):
        self.camera_to_world_pose = camera_to_world_pose
        self.camera_intr = camera_intr
        self.radius = radius
        self.elev = elev
        self.az = az
        self.roll = roll
        self.tx = tx
        self.ty = ty
        self.focal = focal
        self.cx = cx
        self.cy = cy

    @property
    def T_camera_world(self):
        return self.camera_to_world_pose

class RenderSample(object):
    """Struct to encapsulate the results of sampling rendered images from a camera.

    Attributes
    ----------
    renders : dict
        A dictionary mapping perception.RenderMode types to perception.Image classes.
    camera : CameraSample
        The camera sample that produced this render sample.
    """
    def __init__(self, renders, camera):
        self.renders = renders
        self.camera = camera

class ViewsphereDiscretizer(object):

    @staticmethod
    def get_camera_poses(config, frame='world'):
        """Get a list of camera-to-frame poses broken up uniformly about a viewsphere.

        Parameters
        ----------
        config : autolab_core.YamlConfig
            A configuration containing parameters of the random variable.
        frame : str
            The name of the target world frame.

        Notes
        -----
        Required parameters of config are specified in Other Parameters.

        Other Parameters
        ----------------
        radius:               Distance from camera to world origin.
            min : float
            max : float
            n   : int
        azimuth:              Azimuth (angle from x-axis) of camera in degrees.
            min : float
            max : float
            n   : int
        elevation:            Elevation (angle from z-axis) of camera in degrees.
            min : float
            max : float
            n   : int
        roll:                 Roll (angle about view direction) of camera in degrees.
            min : float
            max : float
            n   : int

        Returns
        -------
        list of autolab_core.RigidTransform
            A list of camera-to-frame transforms.
        """
        min_radius = config['radius']['min']
        max_radius = config['radius']['max']
        num_radius = config['radius']['n']
        radii = np.linspace(min_radius, max_radius, num_radius)

        min_azimuth = config['azimuth']['min']
        max_azimuth = config['azimuth']['max']
        num_azimuth = config['azimuth']['n']
        azimuths = np.linspace(min_azimuth, max_azimuth, num_azimuth)

        min_elev = config['elev']['min']
        max_elev = config['elev']['max']
        num_elev = config['elev']['n']
        elevs = np.linspace(min_elev, max_elev, num_elev)

        min_roll = config['roll']['min']
        max_roll = config['roll']['max']
        num_roll = config['roll']['n']
        rolls = np.linspace(min_roll, max_roll, num_roll)

        camera_to_frame_tfs = []
        for r in radii:
            for a in azimuths:
                for e in elevs:
                    for roll in rolls:
                        cam_center = np.array([sph2cart(r, a, e)]).squeeze()
                        cz = -cam_center / np.linalg.norm(cam_center)
                        cx = np.array([cz[1], -cz[0], 0])
                        if np.linalg.norm(cx) == 0:
                            cx = np.array([1.0, 0.0, 0.0])
                        cx = cx / np.linalg.norm(cx)
                        cy = np.cross(cz, cx)
                        cy = cy / np.linalg.norm(cy)
                        if cy[2] > 0:
                            cx = -cx
                            cy = np.cross(cz, cx)
                            cy = cy / np.linalg.norm(cy)
                        R_cam_frame = np.array([cx, cy, cz]).T
                        R_roll = RigidTransform.z_axis_rotation(roll)
                        R_cam_frame = R_cam_frame.dot(R_roll)

                        T_camera_frame = RigidTransform(R_cam_frame, cam_center,
                                                        from_frame='camera', to_frame=frame)
                        camera_to_frame_tfs.append(T_camera_frame)
        return camera_to_frame_tfs

class UniformPlanarWorksurfaceRandomVariable(RandomVariable):
    """Uniform distribution over camera poses and intrinsics about a viewsphere over a planar worksurface.
    The camera is positioned pointing towards (0,0,0).
    """

    def __init__(self, frame, config, num_prealloc_samples=1):
        """Initialize a UniformPlanarWorksurfaceRandomVariable.

        Parameters
        ----------
        frame : str
            string name of the camera frame
        config : autolab_core.YamlConfig
            configuration containing parameters of random variable
        num_prealloc_samples : int
            Number of preallocated samples.

        Notes
        -----
        Required parameters of config are specified in Other Parameters

        Other Parameters
        ----------
        focal_length :        Focal length of the camera
            min : float
            max : float
        delta_optical_center: Change in optical center from neutral.
            min : float
            max : float
        radius:               Distance from camera to world origin.
            min : float
            max : float
        azimuth:              Azimuth (angle from x-axis) of camera in degrees.
            min : float
            max : float
        elevation:            Elevation (angle from z-axis) of camera in degrees.
            min : float
            max : float
        roll:                 Roll (angle about view direction) of camera in degrees.
            min : float
            max : float
        x:                    Translation of world center in x axis.
            min : float
            max : float
        y:                    Translation of world center in y axis.
            min : float
            max : float
        im_height : float     Height of image in pixels.
        im_width : float      Width of image in pixels.
        """
        # read params
        self.frame = frame
        self.config = config
        self.num_prealloc_samples = num_prealloc_samples

        self._parse_config(config)

        # setup random variables

        # camera
        self.focal_rv = ss.uniform(loc=self.min_f, scale=self.max_f-self.min_f)
        self.cx_rv = ss.uniform(loc=self.min_cx, scale=self.max_cx-self.min_cx)
        self.cy_rv = ss.uniform(loc=self.min_cy, scale=self.max_cy-self.min_cy)

        # viewsphere
        self.rad_rv = ss.uniform(loc=self.min_radius, scale=self.max_radius-self.min_radius)
        self.elev_rv = ss.uniform(loc=self.min_elev, scale=self.max_elev-self.min_elev)
        self.az_rv = ss.uniform(loc=self.min_az, scale=self.max_az-self.min_az)
        self.roll_rv = ss.uniform(loc=self.min_roll, scale=self.max_roll-self.min_roll)

        # table translation
        self.tx_rv = ss.uniform(loc=self.min_x, scale=self.max_x-self.min_x)
        self.ty_rv = ss.uniform(loc=self.min_y, scale=self.max_y-self.min_y)

        RandomVariable.__init__(self, self.num_prealloc_samples)

    def _parse_config(self, config):
        """Reads parameters from the config into class members.
        """
        # camera params
        self.min_f = config['focal_length']['min']
        self.max_f = config['focal_length']['max']
        self.min_delta_c = config['delta_optical_center']['min']
        self.max_delta_c = config['delta_optical_center']['max']
        self.im_height = config['im_height']
        self.im_width = config['im_width']

        self.mean_cx = float(self.im_width - 1) / 2
        self.mean_cy = float(self.im_height - 1) / 2
        self.min_cx = self.mean_cx + self.min_delta_c
        self.max_cx = self.mean_cx + self.max_delta_c
        self.min_cy = self.mean_cy + self.min_delta_c
        self.max_cy = self.mean_cy + self.max_delta_c

        # viewsphere params
        self.min_radius = config['radius']['min']
        self.max_radius = config['radius']['max']
        self.min_az = np.deg2rad(config['azimuth']['min'])
        self.max_az = np.deg2rad(config['azimuth']['max'])
        self.min_elev = np.deg2rad(config['elevation']['min'])
        self.max_elev = np.deg2rad(config['elevation']['max'])
        self.min_roll = np.deg2rad(config['roll']['min'])
        self.max_roll = np.deg2rad(config['roll']['max'])

        # params of translation in plane
        self.min_x = config['x']['min']
        self.max_x = config['x']['max']
        self.min_y = config['y']['min']
        self.max_y = config['y']['max']

    def camera_to_world_pose(self, radius, elev, az, roll, x, y):
        """Convert spherical coords to a camera pose in the world.
        """
        # generate camera center from spherical coords
        delta_t = np.array([x, y, 0])
        camera_z = np.array([sph2cart(radius, az, elev)]).squeeze()
        camera_center = camera_z + delta_t
        camera_z = -camera_z / np.linalg.norm(camera_z)

        # find the canonical camera x and y axes
        camera_x = np.array([camera_z[1], -camera_z[0], 0])
        x_norm = np.linalg.norm(camera_x)
        if x_norm == 0:
            camera_x = np.array([1, 0, 0])
        else:
            camera_x = camera_x / x_norm
        camera_y = np.cross(camera_z, camera_x)
        camera_y = camera_y / np.linalg.norm(camera_y)

        # Reverse the x direction if needed so that y points down
        if camera_y[2] > 0:
            camera_x = -camera_x
            camera_y = np.cross(camera_z, camera_x)
            camera_y = camera_y / np.linalg.norm(camera_y)

        # rotate by the roll
        R = np.vstack((camera_x, camera_y, camera_z)).T
        roll_rot_mat = transformations.rotation_matrix(roll, camera_z, np.zeros(3))[:3,:3]
        R = roll_rot_mat.dot(R)
        T_camera_world = RigidTransform(R, camera_center, from_frame=self.frame, to_frame='world')

        return T_camera_world

    def camera_intrinsics(self, T_camera_world, f, cx, cy):
        """Generate shifted camera intrinsics to simulate cropping.
        """
        # form intrinsics
        camera_intr = CameraIntrinsics(self.frame, fx=f, fy=f,
                                       cx=cx, cy=cy, skew=0.0,
                                       height=self.im_height, width=self.im_width)

        return camera_intr

    def sample(self, size=1):
        """Sample random variables from the model.

        Parameters
        ----------
        size : int
            number of sample to take

        Returns
        -------
        :obj:`list` of :obj:`CameraSample`
            sampled camera intrinsics and poses
        """
        samples = []
        for i in range(size):
            # sample camera params
            focal = self.focal_rv.rvs(size=1)[0]
            cx = self.cx_rv.rvs(size=1)[0]
            cy = self.cy_rv.rvs(size=1)[0]

            # sample viewsphere params
            radius = self.rad_rv.rvs(size=1)[0]
            elev = self.elev_rv.rvs(size=1)[0]
            az = self.az_rv.rvs(size=1)[0]
            roll = self.roll_rv.rvs(size=1)[0]

            # sample plane translation
            tx = self.tx_rv.rvs(size=1)[0]
            ty = self.ty_rv.rvs(size=1)[0]

            logging.debug('Sampled')

            logging.debug('focal: %.3f' %(focal))
            logging.debug('cx: %.3f' %(cx))
            logging.debug('cy: %.3f' %(cy))

            logging.debug('radius: %.3f' %(radius))
            logging.debug('elev: %.3f' %(elev))
            logging.debug('az: %.3f' %(az))
            logging.debug('roll: %.3f' %(roll))

            logging.debug('tx: %.3f' %(tx))
            logging.debug('ty: %.3f' %(ty))

            # convert to pose and intrinsics
            T_camera_world = self.camera_to_world_pose(radius, elev, az, roll, tx, ty)
            camera_shifted_intr = self.camera_intrinsics(T_camera_world,
                                                         focal, cx, cy)
            camera_sample = CameraSample(T_camera_world,
                                         camera_shifted_intr,
                                         radius, elev, az, roll, tx=tx, ty=ty,
                                         focal=focal, cx=cx, cy=cy)

            # convert to camera pose
            samples.append(camera_sample)

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples

class UniformPlanarWorksurfaceImageRandomVariable(RandomVariable):
    """Random variable for sampling images from a camera positioned about an object on a table.
    """

    def __init__(self, object_name, scene, render_modes, frame, config, num_prealloc_samples=0):
        """Initialize a UniformPlanarWorksurfaceImageRandomVariable.

        Parameters
        ----------
        object_name : str
            The name of the object to render views about
        scene : Scene
            The scene to be rendered which contains the target object.
        render_modes : list of perception.RenderMode
            A list of RenderModes that indicate the wrapped images to return.
        frame : str
            The name of the camera's frame of reference.
        config : autolab_core.YamlConfig
            A configuration containing parameters of the random variable.
        num_prealloc_samples : int
            Number of preallocated samples.

        Notes
        -----
        Required parameters of config are specified in Other Parameters.

        Other Parameters
        ----------------
        focal_length :        Focal length of the camera
            min : float
            max : float
        delta_optical_center: Change in optical center from neutral.
            min : float
            max : float
        radius:               Distance from camera to world origin.
            min : float
            max : float
        azimuth:              Azimuth (angle from x-axis) of camera in degrees.
            min : float
            max : float
        elevation:            Elevation (angle from z-axis) of camera in degrees.
            min : float
            max : float
        roll:                 Roll (angle about view direction) of camera in degrees.
            min : float
            max : float
        x:                    Translation of world center in x axis.
            min : float
            max : float
        y:                    Translation of world center in y axis.
            min : float
            max : float
        im_height : float     Height of image in pixels.
        im_width : float      Width of image in pixels.
        """
        # read params
        self.object_name = object_name
        self.scene = scene
        self.render_modes = render_modes
        self.frame = frame
        self.config = config
        self.num_prealloc_samples = num_prealloc_samples

        # init random variables
        self.ws_rv = UniformPlanarWorksurfaceRandomVariable(self.frame, self.config, num_prealloc_samples=self.num_prealloc_samples)

        RandomVariable.__init__(self, self.num_prealloc_samples)

    def sample(self, size=1, front_and_back=False):
        """ Sample random variables from the model.

        Parameters
        ----------
        size : int
            Number of samples to take
        front_and_back : bool
            If True, all normals are treated as facing the camera

        Returns
        -------
        list of RenderSample
            A list of samples of renders taken with random camera poses about the scene.
            If size was 1, returns a single sample rather than a list.
        """
        # Save scene's original camera
        orig_camera = self.scene.camera

        obj_xy = np.array(self.scene.objects[self.object_name].T_obj_world.translation)
        obj_xy[2] = 0.0

        samples = []
        for i in range(size):
            # sample camera params
            camera_sample = self.ws_rv.sample(size=1)

            # Compute the camera-to-world transform from the object-to-camera transform
            T_camera_world = camera_sample.camera_to_world_pose
            T_camera_world.translation += obj_xy

            # Set the scene's camera
            camera = VirtualCamera(camera_sample.camera_intr, T_camera_world)
            self.scene.camera = camera

            # Render the scene and grab the appropriate wrapped images
            images = self.scene.wrapped_render(self.render_modes, front_and_back=front_and_back)

            # If a segmask was requested, re-render the scene after disabling all other objects.
            seg_image = None
            if RenderMode.SEGMASK in self.render_modes:
                # Disable every object that isn't the target
                for obj in self.scene.objects.keys():
                    if obj != self.object_name:
                        self.scene.objects[obj].enabled = False

                # Compute the Seg Image
                seg_image = self.scene.wrapped_render([RenderMode.SEGMASK], front_and_back=front_and_back)[0]

                # Re-enable every object
                for obj in self.scene.objects.keys():
                    self.scene.objects[obj].enabled = True

            renders = { m : i for m, i in zip(self.render_modes, images) }
            if seg_image:
                renders[RenderMode.SEGMASK] = seg_image

            samples.append(RenderSample(renders, camera_sample))

        self.scene.camera = orig_camera

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples
