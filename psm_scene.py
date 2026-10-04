"""
psm_scene.py
=============

Modelado de la escena del simulador laparoscópico tipo "peg transfer":
mesa de trabajo, generación aleatoria de cilindros y utilidades de
visualización 3D.

Cada cilindro tiene:
- Posición de su centro geométrico [m] en frame mundo.
- Orientación: matriz de rotación 3x3 que mapea el frame local del
  cilindro (con z_local = eje del cilindro) al frame mundo.
- Radio y altura.

Convenciones del frame del cilindro:
- z_local : eje longitudinal del cilindro (las "tapas" están en z = ±h/2).
- x_local, y_local : ejes radiales arbitrarios (definen el plano de las tapas).

Estado lógico de cada cilindro:
- "free"     : reposando sobre la mesa.
- "grasped"  : sujeto por la pinza (su pose se actualiza con el TCP).
- "placed"   : ya colocado en su destino final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# =============================================================================
# Constantes de la escena
# =============================================================================

# Mesa de trabajo: plano horizontal a una altura fija bajo el fulcro.
TABLE_Z = -0.16          # altura de la superficie de la mesa [m]
TABLE_HALF_X = 0.060     # mesa: rango ±X visible
TABLE_HALF_Y_MIN = -0.13 # mesa: rango Y visible (mín)
TABLE_HALF_Y_MAX = -0.02 # mesa: rango Y visible (máx)

# Zona de aparición aleatoria de cilindros (más restringida que la mesa
# para garantizar alcanzabilidad por el robot).
SPAWN_X_MIN, SPAWN_X_MAX = -0.025, +0.025      # 5 cm de ancho
SPAWN_Y_MIN, SPAWN_Y_MAX = -0.095, -0.045      # 5 cm de profundidad

# Geometría del cilindro (proporcional a un dedo de peg trainer real,
# escalado para que la pinza EndoWrist abra ~10 mm y pueda agarrarlo).
CYL_RADIUS = 0.0035   # 7 mm de diámetro
CYL_HEIGHT = 0.020    # 20 mm de altura

# Distancia mínima entre centros de cilindros para evitar superposición.
MIN_CYL_DIST = 0.025  # 25 mm

# Apertura nominal de la pinza (jaw_dx = -30°, jaw_sx = +30° abierta, 0° cerrada).
GRIPPER_OPEN_RAD = np.radians(-30.0)
GRIPPER_CLOSE_RAD = 0.0


# =============================================================================
# Modelo de cilindro
# =============================================================================

@dataclass
class Cylinder:
    """Cilindro rígido en la escena.

    Attributes
    ----------
    position : np.ndarray, shape (3,)
        Posición del centro geométrico [m] en frame mundo.
    rotation : np.ndarray, shape (3, 3)
        Matriz de rotación de frame local del cilindro a frame mundo.
        El eje z local apunta a lo largo del cilindro.
    radius : float
        Radio del cilindro [m].
    height : float
        Altura (longitud a lo largo del eje del cilindro) [m].
    state : str
        Estado lógico: "free", "grasped" o "placed".
    grasp_offset : np.ndarray, optional
        Cuando está "grasped", almacena la pose relativa del cilindro
        respecto al TCP (4x4) para reproducir la rigidez del agarre.
    label : str
        Identificador legible (ej. "C0", "C1", "C2").
    color : str
        Color de visualización.
    """
    position: np.ndarray
    rotation: np.ndarray
    radius: float = CYL_RADIUS
    height: float = CYL_HEIGHT
    state: str = "free"
    grasp_T_rel: np.ndarray | None = None
    label: str = ""
    color: str = "#1f77b4"

    @property
    def axis(self) -> np.ndarray:
        """Eje longitudinal (z local) en frame mundo."""
        return self.rotation[:, 2]

    @property
    def is_vertical(self) -> bool:
        """True si el eje del cilindro está cerca de la vertical mundo."""
        return abs(self.axis[2]) > 0.95

    def pose_homogeneous(self) -> np.ndarray:
        """Pose 4x4 del cilindro en frame mundo."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T


# =============================================================================
# Generación de cilindros
# =============================================================================

def _random_orientation_for_axis(axis_kind: str, rng: np.random.Generator) -> np.ndarray:
    """Construye una rotación según la orientación del cilindro.

    - "vertical": eje z del cilindro alineado con +Z mundo.
    - "horizontal": eje z del cilindro a lo largo de una dirección horizontal
      aleatoria (vector unitario en el plano XY mundo).
    """
    if axis_kind == "vertical":
        # Eje z mundo = (0,0,1); cualquier rotación XY del cilindro es válida
        roll = rng.uniform(0.0, 2.0 * np.pi)
        ca, sa = np.cos(roll), np.sin(roll)
        R = np.array([
            [ca, -sa, 0.0],
            [sa,  ca, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return R
    elif axis_kind == "horizontal":
        # Eje z del cilindro = dirección horizontal aleatoria
        theta = rng.uniform(0.0, 2.0 * np.pi)
        z_axis = np.array([np.cos(theta), np.sin(theta), 0.0])
        # Construir x_local perpendicular a z, en el plano (z_axis, +Z mundo)
        # = + Z mundo sería el natural pero z_axis es horizontal,
        # así que x_local = +Z mundo es perpendicular automaticamente
        x_axis = np.array([0.0, 0.0, 1.0])
        # y_axis = z x x para completar el triedro derecho
        y_axis = np.cross(z_axis, x_axis)
        # Reordenar a R = [x, y, z]
        R = np.column_stack([x_axis, y_axis, z_axis])
        return R
    else:
        raise ValueError(f"axis_kind desconocido: {axis_kind}")


def generate_random_cylinders(
    n: int = 3,
    rng: np.random.Generator | None = None,
    max_attempts: int = 200,
) -> list[Cylinder]:
    """Genera ``n`` cilindros aleatorios sobre la mesa, no superpuestos.

    Cada cilindro tiene 50% de probabilidad de aparecer **vertical** (apoyado
    en su tapa, eje a lo largo de +Z mundo) o **horizontal** (apoyado de lado,
    eje en el plano XY con dirección aleatoria).

    Parameters
    ----------
    n : int
        Número de cilindros (típicamente 3).
    rng : np.random.Generator, optional
        Generador aleatorio. Si None, se crea uno con seed temporal.
    max_attempts : int
        Intentos máximos para colocar cada cilindro respetando distancias.

    Returns
    -------
    cylinders : list of Cylinder
        Lista de cilindros con estado "free".
    """
    if rng is None:
        rng = np.random.default_rng()

    palette = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    cylinders: list[Cylinder] = []

    for i in range(n):
        # Elegir orientación
        axis_kind = "vertical" if rng.random() < 0.5 else "horizontal"

        # Para que repose sobre la mesa, el centro del cilindro está a
        # h/2 (vertical) o radius (horizontal) sobre la mesa.
        if axis_kind == "vertical":
            z_center = TABLE_Z + CYL_HEIGHT / 2.0
        else:
            z_center = TABLE_Z + CYL_RADIUS

        # Buscar posición no-superpuesta
        placed = False
        for _ in range(max_attempts):
            x = rng.uniform(SPAWN_X_MIN, SPAWN_X_MAX)
            y = rng.uniform(SPAWN_Y_MIN, SPAWN_Y_MAX)
            pos = np.array([x, y, z_center])
            ok = True
            for c in cylinders:
                if np.linalg.norm(pos[:2] - c.position[:2]) < MIN_CYL_DIST:
                    ok = False
                    break
            if ok:
                R = _random_orientation_for_axis(axis_kind, rng)
                cylinders.append(Cylinder(
                    position=pos,
                    rotation=R,
                    label=f"C{i}",
                    color=palette[i % len(palette)],
                ))
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"No se pudo colocar el cilindro {i} sin superposición tras "
                f"{max_attempts} intentos."
            )

    return cylinders


# =============================================================================
# Cálculo de poses de agarre
# =============================================================================

def compute_grasp_pose(
    cyl: Cylinder,
    psm_home_pose: np.ndarray,
) -> np.ndarray:
    """Calcula la pose de agarre del TCP para un cilindro dado.

    Estrategia:
        - El agarre es "lateral": las mordazas se cierran perpendicularmente
          al eje del cilindro, abrazándolo a la altura de su centro.
        - El eje Z del TCP (frente de la pinza, dirección de la "boca")
          apunta hacia el cilindro.
        - El eje Y del TCP (perpendicular a la apertura de las mordazas)
          se alinea con el eje del cilindro, de modo que las mordazas se
          cierren en el plano que contiene al cilindro radialmente.

    Para evitar singularidades del PSM, la dirección de aproximación se
    elige según la orientación del cilindro:
        - Cilindro vertical: aproximación desde el lado donde apunta el
          shaft del PSM (frame del HOME) para minimizar reorientación.
        - Cilindro horizontal: aproximación desde arriba (eje Z del TCP
          apunta hacia -Z mundo), abrazando el cilindro acostado.

    Parameters
    ----------
    cyl : Cylinder
        Cilindro a agarrar.
    psm_home_pose : np.ndarray, shape (4, 4)
        Pose homogénea del TCP en HOME, usada como referencia para elegir
        la dirección de aproximación más eficiente.

    Returns
    -------
    T_grasp : np.ndarray, shape (4, 4)
        Pose objetivo del TCP en el momento del cierre de mordazas.
    """
    cyl_axis = cyl.axis  # eje del cilindro en mundo

def compute_grasp_pose(
    cyl: Cylinder,
    psm_home_pose: np.ndarray,
) -> np.ndarray:
    """Calcula la pose de agarre del TCP para un cilindro dado.

    Garantiza consistencia cinemática de 5 GDL (q6 = 0.0) para paridad
    1:1 con la pinza EndoWrist (4 GDL) de Unity.
    """
    from psm_kinematics import L_PITCH2YAW, L_YAW2CTRL, forward_kinematics

    if psm_home_pose.ndim == 1:
        psm_home_pose = forward_kinematics(psm_home_pose)

    home_pos = psm_home_pose[:3, 3]
    approach_dir_horiz = cyl.position[:2] - home_pos[:2]
    norm_h = np.linalg.norm(approach_dir_horiz)
    if norm_h < 1e-6:
        approach_dir_horiz = np.array([0.0, -1.0])
    else:
        approach_dir_horiz = approach_dir_horiz / norm_h

    # Dirección de aproximación con ángulo de descenso de 35°
    descent_angle = np.radians(35.0)
    z_tcp = np.array([
        approach_dir_horiz[0] * np.cos(descent_angle),
        approach_dir_horiz[1] * np.cos(descent_angle),
        -np.sin(descent_angle),
    ])
    z_tcp = z_tcp / np.linalg.norm(z_tcp)

    # Posición del centro de la muñeca
    p_wrist = cyl.position - (L_PITCH2YAW + L_YAW2CTRL) * z_tcp
    norm_w = np.linalg.norm(p_wrist)
    z_shaft = p_wrist / norm_w if norm_w > 1e-6 else np.array([0.0, 0.0, -1.0])

    # Eje de flexión de la muñeca (y_tcp) perpendicular al plano (z_shaft, z_tcp)
    # Esto garantiza q6 = 0.0 exactamente
    y_tcp = np.cross(z_shaft, z_tcp)
    if np.linalg.norm(y_tcp) < 1e-6:
        y_tcp = np.array([1.0, 0.0, 0.0])
    else:
        y_tcp = y_tcp / np.linalg.norm(y_tcp)

    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp = x_tcp / np.linalg.norm(x_tcp)

    R_tcp = np.column_stack([x_tcp, y_tcp, z_tcp])

    T = np.eye(4)
    T[:3, :3] = R_tcp
    T[:3, 3] = cyl.position.copy()
    return T


def compute_approach_pose(T_grasp: np.ndarray, offset: float = 0.020) -> np.ndarray:
    """Pose pre-agarre: retrocede ``offset`` a lo largo del -Z del TCP.

    Es decir, la pinza se posiciona ``offset`` metros antes del cilindro
    en la dirección desde la cual va a aproximarse.
    """
    T_app = T_grasp.copy()
    z_tcp = T_grasp[:3, 2]
    T_app[:3, 3] = T_grasp[:3, 3] - offset * z_tcp
    return T_app


def compute_lift_pose(T_grasp: np.ndarray, height: float = 0.025) -> np.ndarray:
    """Pose post-agarre: eleva ``height`` metros sobre la pose de agarre.

    "Elevar" en frame mundo = subir +Z, no a lo largo de un eje del TCP.
    """
    T_lift = T_grasp.copy()
    T_lift[:3, 3] = T_grasp[:3, 3] + np.array([0.0, 0.0, height])
    return T_lift


def compute_place_grasp_pose(
    cyl: Cylinder,
    place_xy: np.ndarray,
    psm_home_pose: np.ndarray,
) -> np.ndarray:
    """Pose en la que se libera el cilindro en su destino, vertical.

    El cilindro se coloca apoyado verticalmente sobre la mesa en la posición
    XY indicada. Se usa la misma estrategia de agarre que para un cilindro
    vertical genérico.
    """
    target_pos = np.array([place_xy[0], place_xy[1], TABLE_Z + CYL_HEIGHT / 2.0])
    cyl_target = Cylinder(
        position=target_pos,
        rotation=np.eye(3),  # vertical: eje z mundo
        label=cyl.label,
    )
    return compute_grasp_pose(cyl_target, psm_home_pose), cyl_target.position


# =============================================================================
# Utilidades de geometría del cilindro para visualización
# =============================================================================

def cylinder_mesh(
    cyl: Cylinder,
    n_circ: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera la malla 3D para visualizar un cilindro con plot_surface.

    Returns
    -------
    X, Y, Z : np.ndarray, shape (2, n_circ)
        Mallas XYZ para el cuerpo lateral. Para tapas se pueden usar
        los círculos en z = ±h/2 del frame local.
    """
    h2 = cyl.height / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_circ)
    # Coordenadas locales del cuerpo lateral
    x_local = cyl.radius * np.cos(theta)
    y_local = cyl.radius * np.sin(theta)
    z_local_top = np.full_like(theta, +h2)
    z_local_bot = np.full_like(theta, -h2)

    pts_top_local = np.vstack([x_local, y_local, z_local_top])  # (3, n)
    pts_bot_local = np.vstack([x_local, y_local, z_local_bot])  # (3, n)
    pts_top_world = cyl.rotation @ pts_top_local + cyl.position[:, None]
    pts_bot_world = cyl.rotation @ pts_bot_local + cyl.position[:, None]

    X = np.vstack([pts_bot_world[0], pts_top_world[0]])
    Y = np.vstack([pts_bot_world[1], pts_top_world[1]])
    Z = np.vstack([pts_bot_world[2], pts_top_world[2]])
    return X, Y, Z


__all__ = [
    "Cylinder",
    "TABLE_Z",
    "TABLE_HALF_X",
    "TABLE_HALF_Y_MIN",
    "TABLE_HALF_Y_MAX",
    "SPAWN_X_MIN", "SPAWN_X_MAX",
    "SPAWN_Y_MIN", "SPAWN_Y_MAX",
    "CYL_RADIUS",
    "CYL_HEIGHT",
    "MIN_CYL_DIST",
    "GRIPPER_OPEN_RAD",
    "GRIPPER_CLOSE_RAD",
    "generate_random_cylinders",
    "compute_grasp_pose",
    "compute_approach_pose",
    "compute_lift_pose",
    "compute_place_grasp_pose",
    "cylinder_mesh",
]
