"""
psm_ur5_kinematics.py
======================

Cinemática del brazo industrial FAIRINO FR5 V6.0 que sostiene el trocar
laparoscópico, con utilidades para acoplar la base del PSM al efector del FR5.

(Se conserva el nombre de archivo original por compatibilidad de imports.)

Cadena cinemática conjunta:

    Mundo (origen)
     └─ Base FR5 (origen)
          └─ 6 articulaciones FR5 (DH calibrado)
               └─ TCP FR5 (flange · T_tool) ── sostiene rígidamente ──► trocar
                                                                         │
                                          shaft del PSM                  ▼
                                          (atraviesa el trocar)        fulcro/RCM
                                                                         │
                                                           EndoWrist ────▼
                                                                         │
                                                                       TCP PSM

Convenciones:
    - Frame FR5: DH estándar calibrado contra datos reales del controlador.
      No se aplican correcciones adicionales de base (a diferencia del
      modelo UR5 anterior que usaba FLIP y M_EF).
    - FK: T_base_TCP(q) = T_DH(q) · T_tool
    - IK: se deshace T_tool y se resuelve con ikine_LM + multi-seed.
    - El eje Z del TCP del FR5 apunta a lo largo del trocar, desde el
      mango hacia el fulcro (es decir, "hacia adelante" del trocar).
    - El fulcro está a `dist_trocar` metros en +Z del TCP del FR5.
    - El frame base del PSM se ubica EN el fulcro, con su eje Z apuntando
      en la misma dirección que el shaft sale del trocar.

Se proveen dos modos de configuración inicial:

    init_from_fr5_home(robot, T_tool, q_home)
        -> calcula fulcro a partir del HOME del FR5
    init_from_fulcrum(robot, T_tool, p_fulcrum, n)
        -> resuelve IK del FR5 para llegar a un fulcro deseado
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

# --- Añadir Nueva Cinematica al path ---
_CINE_DIR = str(Path(__file__).resolve().parent / "Nueva Cinematica")
if _CINE_DIR not in sys.path:
    sys.path.insert(0, _CINE_DIR)

# pyrefly: ignore [missing-import]
from fr5_kinematics import fk as _fr5_fk, ik_mejor as _fr5_ik, FR5 as _FR5, rpy_a_R, R_a_rpy


# =============================================================================
# Constantes
# =============================================================================

# Distancias por defecto (configurables)
DEFAULT_DIST_TROCAR = 0.15   # m, TCP FR5 -> fulcro
DEFAULT_DIST_TOTAL  = 0.10   # m, TCP FR5 -> punta del shaft
DIST_TROCAR_RANGE   = (0.05, 0.50)  # m, rango configurable


# =============================================================================
def load_fr5():
    return True, np.eye(4)  # Retornar True como dummy robot object para que no falle el chequeo de None


def fr5_home_default() -> np.ndarray:
    """Configuración HOME por defecto (en radianes)."""
    return np.deg2rad([0, -85, 85, -90, -90, 0])


# =============================================================================
def fr5_fk(robot, T_tool, q: np.ndarray) -> np.ndarray:
    return _fr5_fk(q)


def fr5_ik(
    robot,
    T_tool: np.ndarray,
    T_target: np.ndarray,
    q0: np.ndarray,
    ilimit: int = 500,
    tol: float = 1e-9,
    multi_seed: bool = True,
) -> tuple[np.ndarray, bool, float]:
    """IK del FR5 usando el nuevo solver analítico."""
    solucion = _fr5_ik(T_target, q_semilla=q0, salto_max=None)
    if solucion is not None:
        q_sol = solucion.q
        err_pos = np.linalg.norm(_fr5_fk(q_sol)[:3, 3] - T_target[:3, 3]) * 1000.0
        return q_sol, True, err_pos
    else:
        err_pos = np.linalg.norm(_fr5_fk(q0)[:3, 3] - T_target[:3, 3]) * 1000.0
        return q0.copy(), False, err_pos


# =============================================================================
# Geometría del trocar
# =============================================================================

def fulcrum_from_tcp_pose(
    T_tcp: np.ndarray,
    dist_trocar: float = DEFAULT_DIST_TROCAR,
) -> np.ndarray:
    """Posición del fulcro dada la pose del TCP del FR5 (punta de la pinza).

    El fulcro está a `dist_trocar` metros hacia atrás (-Z) del TCP.
    """
    return T_tcp[:3, 3] - dist_trocar * T_tcp[:3, 2]


def shaft_axis_from_tcp_pose(T_tcp: np.ndarray) -> np.ndarray:
    """Dirección unitaria del shaft (eje Z del TCP)."""
    return T_tcp[:3, 2].copy()


def tcp_pose_from_fulcrum_axis(
    p_fulcrum: np.ndarray,
    shaft_axis: np.ndarray,
    dist_trocar: float = DEFAULT_DIST_TROCAR,
) -> np.ndarray:
    """Construye la pose del TCP dada la posición del fulcro y la dir. del shaft.

    Estrategia:
        - La punta del TCP está a `dist_trocar` a lo largo del shaft (+z).
        - El eje Z del TCP coincide con shaft_axis.
    """
    z = shaft_axis / np.linalg.norm(shaft_axis)

    # La punta del TCP queda dist_trocar metros después del fulcro a lo largo de +z
    p_tcp = p_fulcrum + dist_trocar * z

    # Construir base ortonormal: x_local = un vector perpendicular a z
    # arbitrario, luego y_local = z × x_local
    aux = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, aux); y /= np.linalg.norm(y)
    x = np.cross(y, z)

    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = p_tcp
    return T


# =============================================================================
# Acople FR5 -> PSM: transformación del frame base del PSM en el mundo
# =============================================================================

def psm_base_pose_from_fulcrum(
    p_fulcrum: np.ndarray,
    shaft_axis: np.ndarray,
    q_psm_home: np.ndarray | None = None,
) -> np.ndarray:
    """Pose del frame base del PSM (F0) en el mundo, alineada con el trocar.

    Esta función calcula la pose del frame base del PSM tal que, cuando el
    PSM está en su configuración HOME (`q_psm_home`), el shaft del PSM sale
    del fulcro alineado con `shaft_axis` (es decir, en la misma dirección
    en la que apunta el trocar sostenido por el FR5).

    Esto se logra:
        1. Calculando, en el frame base del PSM, la dirección del shaft en
           HOME (vector del RCM al TCP del PSM en HOME).
        2. Encontrando una rotación rígida que lleve esa dirección a
           `shaft_axis` en el mundo.
        3. Construyendo el frame base del PSM con origen en el fulcro y
           esa rotación.

    Si `q_psm_home` es None se asume el HOME laparoscópico estándar
    (q2 = π/6), que es el que produce trayectorias alcanzables sobre la
    mesa por defecto.

    Parameters
    ----------
    p_fulcrum : np.ndarray, (3,)
        Posición del fulcro en el mundo.
    shaft_axis : np.ndarray, (3,)
        Dirección unitaria del shaft (del fulcro al TCP del PSM) en el mundo.
    q_psm_home : np.ndarray, (7,), optional
        Configuración HOME del PSM. Si None usa Q_HOME_LAPARO.

    Returns
    -------
    T_psm_base : np.ndarray, 4x4
        Pose del frame base del PSM en el mundo.
    """
    from psm_kinematics import Q_HOME_LAPARO, forward_kinematics

    if q_psm_home is None:
        q_psm_home = Q_HOME_LAPARO

    # 1. Dirección del shaft en frame base del PSM en HOME
    T_tcp_local = forward_kinematics(q_psm_home)
    p_tcp_local = T_tcp_local[:3, 3]
    norm_local = np.linalg.norm(p_tcp_local)
    if norm_local < 1e-9:
        raise ValueError("TCP del PSM en HOME coincide con el RCM; q3 debe ser > 0.")
    n_local = p_tcp_local / norm_local  # vector unitario en frame PSM

    # 2. Dirección deseada en mundo
    n_world = shaft_axis / np.linalg.norm(shaft_axis)

    # 3. Construir rotación R tal que R @ n_local = n_world
    # Algoritmo de Rodrigues para rotación entre dos vectores
    v = np.cross(n_local, n_world)
    s = np.linalg.norm(v)
    c = np.dot(n_local, n_world)

    if s < 1e-9:
        # Ya están alineados (c ≈ 1) o opuestos (c ≈ -1)
        if c > 0:
            R_align = np.eye(3)
        else:
            # Vectores opuestos: rotar 180° en cualquier eje perpendicular a n_local
            aux = np.array([1.0, 0.0, 0.0]) if abs(n_local[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(n_local, aux); axis /= np.linalg.norm(axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R_align = np.eye(3) + 2.0 * K @ K  # sin(π)=0, (1-cos(π))=2
    else:
        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R_align = np.eye(3) + K + K @ K * ((1 - c) / (s * s))

    T = np.eye(4)
    T[:3, :3] = R_align
    T[:3, 3]  = p_fulcrum
    return T


def transform_psm_pose_to_world(
    T_psm: np.ndarray,
    T_psm_base_world: np.ndarray,
) -> np.ndarray:
    """Transforma una pose en frame PSM al frame mundo.

    Si una pose del PSM (TCP, frame intermedio, cilindro, etc.) está dada
    en el frame base del PSM, esta función la lleva al mundo aplicando la
    transformación rígida del frame base del PSM.
    """
    return T_psm_base_world @ T_psm


# =============================================================================
# Cálculo del FR5 que sostiene el trocar dada la configuración del PSM
# =============================================================================

def fr5_target_from_psm_state(
    psm_q: np.ndarray,
    p_fulcrum_world: np.ndarray,
    T_psm_base_world: np.ndarray,
    dist_trocar: float = DEFAULT_DIST_TROCAR,
) -> np.ndarray:
    """Pose del TCP FR5 coherente con la configuración del PSM.

    Dada una configuración articular del PSM y la pose del frame base del
    PSM en el mundo, calcula la pose que debe tener el TCP del FR5 para
    sostener el trocar alineado con el shaft del PSM en su configuración
    actual.

    El shaft del PSM siempre pasa por el frame base (RCM) del PSM por
    construcción. Su dirección está dada por el eje Z del frame F4 del PSM
    (la base del shaft después de q1, q2, q3, donde q3 es la inserción).

    Parameters
    ----------
    psm_q : np.ndarray, (7,)
        Configuración articular del PSM.
    p_fulcrum_world : np.ndarray, (3,)
        Posición del fulcro en el mundo (= origen del frame base del PSM).
    T_psm_base_world : np.ndarray, 4x4
        Pose del frame base del PSM en el mundo.
    dist_trocar : float
        Distancia TCP FR5 -> fulcro [m].

    Returns
    -------
    T_tcp_target : np.ndarray, 4x4
        Pose objetivo del TCP del FR5 en el frame mundo.
    """
    from psm_kinematics import forward_kinematics_full

    # Frames del PSM en su propio frame base
    _, frames_psm = forward_kinematics_full(psm_q)
    # El shaft del PSM va desde el RCM (frame F0) hacia el TCP (frame F7).
    # Para tomar la dirección del shaft DENTRO del trocar usamos la línea
    # que pasa por el RCM y va hacia el frame F4 (el final del shaft).
    p_tcp_psm_local = frames_psm[7][:3, 3]

    # Transformar al mundo
    p_tcp_psm_world = T_psm_base_world[:3, :3] @ p_tcp_psm_local + T_psm_base_world[:3, 3]

    # Dirección del shaft en el mundo: del fulcro hacia el TCP del PSM
    direction = p_tcp_psm_world - p_fulcrum_world
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        shaft_axis = T_psm_base_world[:3, 2]
    else:
        shaft_axis = direction / norm

    # El robot FR5 TCP ya incluye la pinza completa (D6 = 0.457 m)
    L_tool = 0.0
    # p_tcp_psm_world es la punta real deseada. 
    # El FR5 TCP debe estar L_tool detrás de la punta a lo largo del shaft
    p_tcp_target = p_tcp_psm_world - L_tool * shaft_axis
    
    # Para reusar tcp_pose_from_fulcrum_axis, dist_trocar es la distancia TCP -> fulcro
    dist_trocar_dinamico = np.linalg.norm(p_fulcrum_world - p_tcp_target)
    
    T_tcp = tcp_pose_from_fulcrum_axis(
        p_fulcrum_world, shaft_axis, dist_trocar=dist_trocar_dinamico,
    )
    return T_tcp


# =============================================================================
# Modos de inicialización
# =============================================================================

def init_from_fr5_home(
    robot: rtb.DHRobot,
    T_tool: np.ndarray,
    q_home: np.ndarray | None = None,
    dist_trocar: float = DEFAULT_DIST_TROCAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Modo de inicialización 1: el HOME del FR5 define el fulcro.

    Returns
    -------
    q_home : np.ndarray, (6,)
        Configuración HOME utilizada.
    T_tcp : np.ndarray, 4x4
        Pose del TCP del FR5 en HOME.
    p_fulcrum_world : np.ndarray, (3,)
        Posición del fulcro en el mundo.
    T_psm_base_world : np.ndarray, 4x4
        Pose del frame base del PSM en el mundo.
    """
    if q_home is None:
        q_home = fr5_home_default()
    T_tcp = fr5_fk(robot, T_tool, q_home)
    p_fulcrum = fulcrum_from_tcp_pose(T_tcp, dist_trocar=dist_trocar)
    shaft_axis = shaft_axis_from_tcp_pose(T_tcp)
    T_psm_base = psm_base_pose_from_fulcrum(p_fulcrum, shaft_axis)
    return q_home, T_tcp, p_fulcrum, T_psm_base


def init_from_fulcrum(
    robot: rtb.DHRobot,
    T_tool: np.ndarray,
    p_fulcrum_world: np.ndarray,
    shaft_axis: np.ndarray,
    q_seed: np.ndarray | None = None,
    dist_trocar: float = DEFAULT_DIST_TROCAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Modo de inicialización 2: el usuario define dónde queda el fulcro.

    Resuelve IK del FR5 para llegar a la pose del trocar requerida por
    `(p_fulcrum_world, shaft_axis)`.

    Returns
    -------
    q : np.ndarray, (6,)
        Configuración del FR5 obtenida.
    T_tcp : np.ndarray, 4x4
        Pose efectiva del TCP del FR5 (después de FK del q resuelto).
    p_fulcrum_world : np.ndarray, (3,)
        Posición del fulcro (recalculada con la pose efectiva).
    T_psm_base_world : np.ndarray, 4x4
        Pose del frame base del PSM en el mundo.
    success : bool
    err_mm : float
    """
    if q_seed is None:
        q_seed = fr5_home_default()
    T_target = tcp_pose_from_fulcrum_axis(
        p_fulcrum_world, shaft_axis, dist_trocar=dist_trocar,
    )
    q_sol, ok, err = fr5_ik(robot, T_tool, T_target, q_seed)
    T_actual = fr5_fk(robot, T_tool, q_sol)
    # Recalcular fulcro real con la pose efectiva
    p_fulc_actual = fulcrum_from_tcp_pose(T_actual, dist_trocar=dist_trocar)
    shaft_actual  = shaft_axis_from_tcp_pose(T_actual)
    T_psm_base = psm_base_pose_from_fulcrum(p_fulc_actual, shaft_actual)
    return q_sol, T_actual, p_fulc_actual, T_psm_base, ok, err


__all__ = [
    "DEFAULT_DIST_TROCAR", "DEFAULT_DIST_TOTAL", "DIST_TROCAR_RANGE",
    "load_fr5", "fr5_home_default",
    "fr5_fk", "fr5_ik",
    "fulcrum_from_tcp_pose", "shaft_axis_from_tcp_pose",
    "tcp_pose_from_fulcrum_axis",
    "psm_base_pose_from_fulcrum", "transform_psm_pose_to_world",
    "fr5_target_from_psm_state",
    "init_from_fr5_home", "init_from_fulcrum",
]
