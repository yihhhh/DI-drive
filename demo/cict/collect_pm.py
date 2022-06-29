import numpy as np
import cv2
import carla
import os, sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(os.path.join(SCRIPT_DIR, "camera"))
# sys.path.append(os.path.join(SCRIPT_DIR, "camera"))
# print(sys.path)
# import camera
# # from camera import camera.CameraParams, camera.IntrinsicParams, camera.ExtrinsicParams
# # from camera import camera.CoordinateTransformation, camera.rotationMatrix3D

def rotationMatrix3D(roll, pitch, yaw):
    # RPY <--> XYZ, roll first, picth then, yaw final
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(3)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def rotationMatrixRoll(roll):
    R = np.identity(3)
    R[1, 1] = np.cos(roll)
    R[2, 2] = np.cos(roll)
    R[2, 1] = np.sin(roll)
    R[1, 2] = -np.sin(roll)
    return R


def rotarotationMatrixPitch(pitch):
    R = np.identity(3)
    R[0, 0] = np.cos(pitch)
    R[2, 2] = np.cos(pitch)
    R[2, 0] = -np.sin(pitch)
    R[0, 2] = np.sin(pitch)
    return R


def rotarotationMatrixYaw(yaw):
    R = np.identity(3)
    R[0, 0] = np.cos(yaw)
    R[1, 1] = np.cos(yaw)
    R[1, 0] = np.sin(yaw)
    R[0, 1] = -np.sin(yaw)
    return R


def rotationMatrix3DYPR(roll, pitch, yaw):
    return np.dot(np.dot(rotationMatrixRoll(roll), rotarotationMatrixPitch(pitch)), rotarotationMatrixYaw(yaw))


def reverseX():
    I = np.identity(3)
    I[0, 0] = -1
    return I


def reverseY():
    I = np.identity(3)
    I[1, 1] = -1
    return I


def intrinsicMatrix(fx, fy, u0, v0):
    K = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    return K


class CoordinateTransformation(object):
    I = np.dot(np.dot(reverseX(), reverseY()), rotationMatrix3DYPR(np.pi / 2, 0, -np.pi / 2))

    @staticmethod
    def world3DToCamera3D(world_vec, R, t):
        camera_vec = np.dot(R.T, world_vec - t)
        return camera_vec

    @staticmethod
    def camera3DToWorld3D(camera_vec, R, t):
        world_vec = np.dot(R, camera_vec) + t
        return world_vec

    @staticmethod
    def camera3DToImage2D(camera_vec, K, eps=1e-24):
        image_vec = np.dot(np.dot(K, CoordinateTransformation.I), camera_vec)
        return image_vec[:2, :] / (image_vec[2, :] + eps)

    @staticmethod
    def world3DToImage2D(world_vec, K, R, t):
        camera_vec = CoordinateTransformation.world3DToCamera3D(world_vec, R, t)
        image_vec = CoordinateTransformation.camera3DToImage2D(camera_vec, K)
        return image_vec

    @staticmethod
    def world3DToImagePixel2D(world_vec, K, R, t):
        image_vec = CoordinateTransformation.world3DToImage2D(world_vec, K, R, t)
        x_pixel, y_pixel = round(image_vec[0, 0]), round(image_vec[1, 0])
        return np.array([x_pixel, y_pixel]).reshape(2, 1)

    @staticmethod
    def image2DToWorld3D(image_vec, K, R, t):
        r = np.vstack((image_vec, 1))
        b = np.vstack((np.dot(np.dot(K, CoordinateTransformation.I), t), 0))

        temp1 = np.dot(np.dot(K, CoordinateTransformation.I), R.T)
        temp2 = np.hstack((temp1, -r))
        A = np.vstack((temp2, np.array([[0, 0, 1, 0]])))
        world_vec = np.dot(np.linalg.inv(A), b)
        return world_vec[:3]

    @staticmethod
    def image2DToWorld3D2(image_vec, K, R, t):
        r = np.vstack((image_vec, np.ones((1, image_vec.shape[1]))))
        b = np.vstack((np.dot(np.dot(K, CoordinateTransformation.I), t), 0))

        temp1 = np.dot(np.dot(K, CoordinateTransformation.I), R.T)
        temp1 = np.expand_dims(temp1, axis=2).repeat(image_vec.shape[1], axis=2)
        r = np.expand_dims(r, axis=1)

        temp1 = np.transpose(temp1, (2, 0, 1))
        r = np.transpose(r, (2, 0, 1))

        temp2 = np.concatenate((temp1, -r), axis=2)
        temp3 = np.array([[0, 0, 1, 0]])
        temp3 = np.expand_dims(temp3, axis=2).repeat(image_vec.shape[1], axis=2)
        temp3 = np.transpose(temp3, (2, 0, 1))

        A = np.concatenate((temp2, temp3), axis=1)
        world_vec = np.dot(np.linalg.inv(A), b)
        return world_vec[:, :3]


class IntrinsicParams(object):

    def __init__(self, sensor):
        '''
        Args:
            sensor: carla.Sensor
        '''
        image_size_x = float(sensor.attributes['image_size_x'])
        image_size_y = float(sensor.attributes['image_size_y'])
        fov = eval(sensor.attributes['fov'])
        f = image_size_x / (2 * np.tan(fov * np.pi / 360))

        # [px]
        self.fx = f
        self.fy = f
        self.u0 = image_size_x / 2
        self.v0 = image_size_y / 2


class ExtrinsicParams(object):

    def __init__(self, sensor):
        '''
        Args:
            sensor: carla.Sensor
        '''

        # camera coordinate in world coordinate
        transform = sensor.get_transform()

        # [m]
        self.x = transform.location.x
        self.y = transform.location.y
        self.z = transform.location.z

        # [rad]
        self.roll = np.deg2rad(transform.rotation.roll)
        self.pitch = np.deg2rad(transform.rotation.pitch)
        self.yaw = np.deg2rad(transform.rotation.yaw)


class CameraParams(object):

    def __init__(self, intrinsic_params, extrinsic_params):
        '''
        Args:
            intrinsic_params: IntrinsicParams
            extrinsic_params: ExtrinsicParams
        '''

        self.K = intrinsicMatrix(intrinsic_params.fx, intrinsic_params.fy, intrinsic_params.u0, intrinsic_params.v0)

        # (coordinate) t: camera in world, R: camera to world
        self.t = np.array([[extrinsic_params.x, extrinsic_params.y, extrinsic_params.z]]).T
        self.R = rotationMatrix3D(extrinsic_params.roll, extrinsic_params.pitch, extrinsic_params.yaw)



def rad_lim(rad):
    while (rad > np.pi):
        rad -= (2 * np.pi)
    while (rad < -np.pi):
        rad += (2 * np.pi)
    return rad


def getLinearPose(pose1, pose2, min_dist):
    x1, x2 = pose1.location.x, pose2.location.x
    y1, y2 = pose1.location.y, pose2.location.y
    z1, z2 = pose1.location.z, pose2.location.z
    roll1, roll2 = np.deg2rad(pose1.rotation.roll), np.deg2rad(pose2.rotation.roll)
    pitch1, pitch2, = np.deg2rad(pose1.rotation.pitch), np.deg2rad(pose2.rotation.pitch)
    yaw1, yaw2, = np.deg2rad(pose1.rotation.yaw), np.deg2rad(pose2.rotation.yaw)

    distance = pose1.location.distance(pose2.location)
    total = int(distance / min_dist)
    result_list = []

    tt = np.arange(total) / total
    x, y, z = tt * x2 + (1 - tt) * x1, tt * y2 + (1 - tt) * y1, tt * z2 + (1 - tt) * z1
    roll = np.rad2deg(rad_lim(roll2 - roll1) * tt + roll1)
    pitch = np.rad2deg(rad_lim(pitch2 - pitch1) * tt + pitch1)
    yaw = np.rad2deg(rad_lim(yaw2 - yaw1) * tt + yaw1)

    for i in range(total):
        location = carla.Location(x=x[i], y=y[i], z=z[i])
        rotation = carla.Rotation(roll=roll[i], pitch=pitch[i], yaw=yaw[i])
        result_list.append(carla.Transform(location, rotation))
    return result_list


class CollectPerspectiveImage(object):

    def __init__(self, param, sensor):
        self.longitudinal_sample_number_near = param.longitudinal_sample_number_near
        self.longitudinal_sample_number_far = param.longitudinal_sample_number_far

        self.vehicle_half_width = param.vehicle_width / 2

        self.lateral_step_factor = param.lateral_step_factor
        self.lateral_sample_array = np.linspace(
            -self.vehicle_half_width, self.vehicle_half_width, param.lateral_sample_number
        )

        self.sensor = sensor
        self.camera_params = CameraParams(IntrinsicParams(sensor), ExtrinsicParams(sensor))
        self.img_width = eval(sensor.attributes['image_size_x'])
        self.img_height = eval(sensor.attributes['image_size_y'])
        self.max_pixel = np.array([self.img_height, self.img_width]).reshape([2, 1])
        self.min_pixel = np.zeros((2, 1))

        self.empty_image = np.zeros((self.img_height // 2, self.img_width // 2), dtype=np.dtype("uint8"))

    def data_augmentation(self, traj_pose_list):
        result_list = []
        for i in range(len(traj_pose_list) - 1):
            p1 = traj_pose_list[i][1]
            p2 = traj_pose_list[i + 1][1]
            if float(i) / len(traj_pose_list) < 0.4:
                min_dist = 0.04
            elif float(i) / len(traj_pose_list) < 0.6:
                min_dist = 0.08
            else:
                min_dist = 0.12

            result_list.extend(getLinearPose(p1, p2, min_dist))

        return result_list

    def drawDestInImage(self, dest_vec, location, rotation):
        empty_image = np.zeros((self.img_height // 2, self.img_width // 2, 3), dtype=np.dtype("uint8"))
        R = rotationMatrix3D(np.deg2rad(rotation[2]), np.deg2rad(rotation[0]), np.deg2rad(rotation[1]))
        t = location.reshape(3, 1)
        vehicle_vec = CoordinateTransformation.world3DToCamera3D(dest_vec, R, t)
        pixel_vec = CoordinateTransformation.world3DToImage2D(
            vehicle_vec, self.camera_params.K, self.camera_params.R, self.camera_params.t
        )
        pixel_vec = pixel_vec[::-1, :]
        x_pixel = pixel_vec.astype(int)[0, 0]
        y_pixel = pixel_vec.astype(int)[1, 0]
        #print(dest_vec,pixel_vec)
        x_pixel = np.clip(x_pixel, 10, self.img_height - 10)
        y_pixel = np.clip(y_pixel, 10, self.img_width - 10)
        x_pixel, y_pixel = np.meshgrid(
            np.arange(max(0, x_pixel // 2 - 5), min(self.img_height // 2 - 1, x_pixel // 2 + 5)),
            np.arange(max(0, y_pixel // 2 - 5), min(self.img_width // 2 - 1, y_pixel // 2 + 5)),
            indexing='ij'
        )
        empty_image[x_pixel, y_pixel, 2] = 255

        return cv2.resize(empty_image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)

    def drawLineInImage(self, traj_pose, vehicle_transform):
        #traj_position = traj_pose.location
        traj_vec = np.array([traj_pose.location.x, traj_pose.location.y, traj_pose.location.z]).reshape(3, 1)
        rotation = vehicle_transform.rotation
        location = vehicle_transform.location
        R = rotationMatrix3D(np.deg2rad(rotation.roll), np.deg2rad(rotation.pitch), np.deg2rad(rotation.yaw))
        t = np.array([location.x, location.y, location.z]).reshape(3, 1)

        # along lateral
        theta = np.deg2rad(traj_pose.rotation.yaw + 90)
        start_vec = np.array([self.vehicle_half_width * np.cos(theta), self.vehicle_half_width * np.sin(theta), 0]
                             ).reshape(3, 1) + traj_vec
        start_vehicle_vec = CoordinateTransformation.world3DToCamera3D(start_vec, R, t)
        start_pixel_vec = CoordinateTransformation.world3DToImage2D(
            start_vehicle_vec, self.camera_params.K, self.camera_params.R, self.camera_params.t
        )
        start_pixel_vec = start_pixel_vec[::-1, :]

        theta = np.deg2rad(traj_pose.rotation.yaw - 90)
        end_vec = np.array([self.vehicle_half_width * np.cos(theta), self.vehicle_half_width * np.sin(theta), 0]
                           ).reshape(3, 1) + traj_vec
        end_vehicle_vec = CoordinateTransformation.world3DToCamera3D(end_vec, R, t)
        end_pixel_vec = CoordinateTransformation.world3DToImage2D(
            end_vehicle_vec, self.camera_params.K, self.camera_params.R, self.camera_params.t
        )
        end_pixel_vec = end_pixel_vec[::-1, :]

        flag1 = (start_pixel_vec >= self.min_pixel).all() and (start_pixel_vec < self.max_pixel).all()
        flag2 = (end_pixel_vec >= self.min_pixel).all() and (end_pixel_vec < self.max_pixel).all()
        if not flag1 and not flag2:
            return

        length = np.linalg.norm(end_pixel_vec - start_pixel_vec)
        direction = (end_pixel_vec - start_pixel_vec) / length
        lateral_sample_number = round(length / self.lateral_step_factor) + 1
        distance_array = np.linspace(0, length, lateral_sample_number)

        pixel_vec = start_pixel_vec + distance_array * direction

        x_pixel = pixel_vec.astype(int)[0]
        y_pixel = pixel_vec.astype(int)[1]

        mask = np.where((x_pixel >= 0) & (x_pixel < self.img_height))[0]
        x_pixel = x_pixel[mask]
        y_pixel = y_pixel[mask]

        mask = np.where((y_pixel >= 0) & (y_pixel < self.img_width))[0]
        x_pixel = x_pixel[mask]
        y_pixel = y_pixel[mask]
        self.empty_image[x_pixel // 2, y_pixel // 2] = 255
        self.empty_image[np.clip(x_pixel // 2 + 1, 0, self.img_height // 2 - 1), y_pixel // 2] = 255
        self.empty_image[np.max(x_pixel // 2 - 1, 0), y_pixel // 2] = 255
        return

    def getPM(self, traj_pose_list, vehicle_transform):
        self.empty_image = np.zeros((self.img_height // 2, self.img_width // 2), dtype=np.dtype("uint8"))
        aug_traj_pose_list = self.data_augmentation(traj_pose_list)
        for traj_pose in aug_traj_pose_list:
            self.drawLineInImage(traj_pose, vehicle_transform)

        kernel = np.ones((
            5,
            5,
        ), np.uint8)
        self.empty_image = cv2.dilate(self.empty_image, kernel, iterations=1)
        self.empty_image = cv2.erode(self.empty_image, kernel, iterations=1)
        return cv2.resize(self.empty_image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)


class InversePerspectiveMapping(object):

    def __init__(self, param, sensor):
        self.sensor = sensor
        self.camera_params = CameraParams(IntrinsicParams(sensor), ExtrinsicParams(sensor))

        self.img_width = 400
        self.img_height = 200
        self.empty_image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        self.longutudinal_length = param.longitudinal_length
        self.ksize = param.ksize

        f = float(self.img_height) / self.longutudinal_length
        self.pesudo_K = np.array([[f, 0, self.img_width / 2], [0, f, self.img_height], [0, 0, 1]])
        self.reverseXY = rotationMatrix3D(0, 0, -np.pi / 2)

    def getIPM(self, image):
        self.empty_image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        index_array = np.argwhere(image > 200)
        index_array = index_array[:, :2]
        index_array = np.unique(index_array, axis=0)
        index_array = np.array([index_array[:, 1], index_array[:, 0]])
        vehicle_vec = CoordinateTransformation.image2DToWorld3D2(
            index_array, self.camera_params.K, self.camera_params.R, self.camera_params.t
        )

        vehicle_vec[:, 2, 0] = 1.0
        temp = np.dot(self.pesudo_K, self.reverseXY)
        vehicle_vec = np.squeeze(vehicle_vec, axis=2)
        new_image_vec = np.dot(temp, vehicle_vec.T)
        new_image_vec = new_image_vec[:2, :]
        new_image_vec = new_image_vec[::-1, :]

        new_image_y_pixel = new_image_vec[0, :].astype(int)
        new_image_x_pixel = new_image_vec[1, :].astype(int)

        #self.empty_image[new_image_y_pixel, new_image_x_pixel] = 255

        mask = np.where((new_image_x_pixel >= 0) & (new_image_x_pixel < self.img_width))[0]
        new_image_x_pixel = new_image_x_pixel[mask]
        new_image_y_pixel = new_image_y_pixel[mask]

        mask = np.where((new_image_y_pixel >= 0) & (new_image_y_pixel < self.img_height))[0]
        new_image_x_pixel = new_image_x_pixel[mask]
        new_image_y_pixel = new_image_y_pixel[mask]
        self.empty_image[new_image_y_pixel, new_image_x_pixel] = 255

        self.empty_image[np.clip(new_image_y_pixel + 1, 0, self.img_height - 1), new_image_x_pixel] = 255
        self.empty_image[np.clip(new_image_y_pixel - 1, 0, self.img_height - 1), new_image_x_pixel] = 255

        #self.empty_image = cv2.GaussianBlur(self.empty_image, (self.ksize, self.ksize), 25)
        return self.empty_image

    def get_cost_map(self, ipm, lidar):
        lidar = -lidar
        mask = np.where((lidar[:, 0] > 1.2) | (lidar[:, 0] < -1.2) | (lidar[:, 1] > 2.0) | (lidar[:, 1] < -4.0))[0]
        lidar = lidar[mask, :]
        mask = np.where(lidar[:, 2] > -1.95)[0]
        lidar = lidar[mask, :]
        img2 = np.zeros((self.img_height, self.img_width), np.uint8)
        img2.fill(255)

        pixel_per_meter = float(self.img_height) / self.longutudinal_length
        u = (self.img_height - lidar[:, 1] * pixel_per_meter).astype(int)
        v = (-lidar[:, 0] * pixel_per_meter + self.img_width // 2).astype(int)

        mask = np.where((u >= 0) & (u < self.img_height))[0]
        u = u[mask]
        v = v[mask]

        mask = np.where((v >= 0) & (v < self.img_width))[0]
        u = u[mask]
        v = v[mask]

        img2[u, v] = 0
        #print(u,v)

        kernel = np.ones((17, 17), np.uint8)
        img2 = cv2.erode(img2, kernel, iterations=1)

        kernel_size = (3, 3)
        img = cv2.dilate(ipm, kernel_size, iterations=3)

        img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)

        mask = np.where((img2 < 50))
        u = mask[0]
        v = mask[1]
        img[u, v] = 0
        #kernel_size = (17, 17)
        #kernel_size = (9, 9)
        #sigma = 9#21
        #img = cv2.GaussianBlur(img, kernel_size, sigma)
        return img
