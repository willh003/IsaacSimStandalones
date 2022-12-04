import math
import numpy as np

def euler_of_quat(quats):
    x = quats[0]
    y = quats[1]
    z = quats[2]
    w = quats[3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1) * 180 / math.pi
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2) * 180 / math.pi
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4) * 180 / math.pi
    

    return roll_x, pitch_y, yaw_z # in degrees

def rot_matrix_of_euler(xtheta, ytheta, ztheta):

    c1 = np.cos(xtheta * np.pi / 180)
    s1 = np.sin(xtheta * np.pi / 180)
    c2 = np.cos(ytheta * np.pi / 180)
    s2 = np.sin(ytheta * np.pi / 180)
    c3 = np.cos(ztheta * np.pi / 180)
    s3 = np.sin(ztheta * np.pi / 180)

    matrix=np.array([[c2*c3, -c2*s3, s2],
                [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    
    return matrix

def quat_of_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return (qx, qy, qz, qw)

def is_standing(mem_queue):

    mem = list(mem_queue)

    if len(mem) < 20:
        return True

    for l in mem:
        qx, qy, qz, qw = l[3], l[4], l[5], l[6]

        gz = qx*qx - qy*qy - qz*qz + qw*qw

        if gz > 0:
            return True

    return False

def is_moving(tolerance, mem_queue):
    """
    True if the robot's x, y, or z position has changed by more than 
    [tolerance] in the last 20 physics steps
    """
    mem = list(mem_queue)

    if len(mem) < 20:
        return True

    x0, y0, z0 = mem[0][0], mem[0][1], mem[0][2]
    for l in mem:
        x, y, z = l[0], l[1], l[2]

        if abs(x-x0) > tolerance or abs(y-y0) > tolerance or abs(z-z0) > tolerance:
            return True

    return False


def get_euclidean_distance(state, target):
    # print('target: ')
    # print(target)
    # print("state: ")
    # print(state)
    return math.sqrt((target[0] - state[0]) ** 2 + (target[1] - state[1]) ** 2)

def get_reward(state, target, step):
    reward = 1 / get_euclidean_distance(state, target)

    return reward


def target_achieved(state, target, tolerance):
    return get_euclidean_distance(state, target) < tolerance