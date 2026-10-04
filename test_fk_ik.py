import numpy as np
from CinematicaRobot.fr5_kinematics import fk, ik_mejor, FR5
q_home = np.deg2rad([0, -90, -90, -90, 90, 0])
T = fk(q_home)
print("FK:", T[:3, 3])
q_sol, ramas = ik_mejor(T, q_home)
print("IK sol:", q_sol)
