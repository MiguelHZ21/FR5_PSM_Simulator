"""
psm_pick_place.py
==================

Planeador de trayectorias pick-and-place para el simulador laparoscópico.

Para cada cilindro genera una secuencia de poses (waypoints) que cubre
las 7 fases canónicas del agarre y transferencia:

    1. APPROACH : ir a la pose pre-agarre, mordazas abiertas
    2. GRASP    : descender a la pose de agarre (mordazas siguen abiertas)
    3. CLOSE    : cerrar mordazas (q7 abierto -> 0) en sitio
    4. LIFT     : elevar el cilindro 2.5 cm
    5. TRANSIT  : trasladar al punto de colocación (manteniendo elevación)
    6. LOWER    : descender hasta la pose de liberación
    7. RELEASE  : abrir mordazas (cilindro queda en su nueva posición)
    8. RETRACT  : elevar para liberar la pinza del cilindro colocado

Cada fase está representada por un sub-segmento de waypoints interpolados.
El resultado es una matriz Q de articulaciones a lo largo de toda la
secuencia, junto con metadata por waypoint (fase, índice de cilindro,
estado del cilindro: free/grasped/placed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from psm_kinematics import (
    forward_kinematics,
    inverse_kinematics,
    Q_HOME_LAPARO,
)
from psm_scene import (
    Cylinder,
    compute_grasp_pose,
    compute_approach_pose,
    compute_lift_pose,
    compute_place_grasp_pose,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD,
    TABLE_Z,
    CYL_HEIGHT,
)


Phase = Literal[
    "approach", "grasp", "close",
    "lift", "transit", "lower",
    "release", "retract", "home",
]


@dataclass
class WaypointInfo:
    """Metadata de un waypoint individual."""
    phase: Phase
    cylinder_idx: int  # -1 si no aplica (movimiento de retorno a HOME)
    gripper: float     # apertura q7 [rad]
    cylinder_attached: bool  # True si el cilindro acompaña al TCP en este wp
    T_target: np.ndarray
    # Pose relativa cilindro->TCP en el momento del cierre. Solo significativa
    # cuando cylinder_attached=True para fases lift/transit/lower
    grasp_T_rel: np.ndarray | None = None


@dataclass
class PickPlacePlan:
    """Plan completo de pick-and-place para una escena."""
    Q: np.ndarray                         # (N, 7) articulaciones por waypoint
    waypoints: list[WaypointInfo]         # (N,) metadata por waypoint
    # Pose final de cada cilindro tras la ejecución completa.
    final_cylinder_poses: list[np.ndarray]
    # Sub-índices: dict cyl_idx -> dict phase -> (start, end) inclusive
    phase_indices: dict[int, dict[Phase, tuple[int, int]]]


# =============================================================================
# Interpolación de poses
# =============================================================================

def _slerp_R(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Interpolación esférica entre dos matrices de rotación."""
    R_rel = R0.T @ R1
    # log(R_rel) -> vector axis-angle
    cos_t = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_t)
    if angle < 1e-9:
        return R0 @ R_rel  # ≈ R1
    axis = (1.0 / (2.0 * np.sin(angle))) * np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ])
    a = t * angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_step = np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)
    return R0 @ R_step


def interpolate_poses(
    T0: np.ndarray, T1: np.ndarray, n_steps: int,
) -> list[np.ndarray]:
    """Interpola poses cartesianas linealmente en posición y SLERP en rotación."""
    p0, p1 = T0[:3, 3], T1[:3, 3]
    R0, R1 = T0[:3, :3], T1[:3, :3]
    poses = []
    for i in range(n_steps):
        t = i / max(1, n_steps - 1)
        T = np.eye(4)
        T[:3, 3] = (1.0 - t) * p0 + t * p1
        T[:3, :3] = _slerp_R(R0, R1, t)
        poses.append(T)
    return poses


def stationary_poses(T: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Repite la misma pose n_steps veces (para fases close/release)."""
    return [T.copy() for _ in range(n_steps)]


# =============================================================================
# Planeación
# =============================================================================

@dataclass
class PhaseConfig:
    """Configuración de número de waypoints por fase."""
    approach: int = 14
    grasp: int = 8
    close: int = 6
    lift: int = 8
    transit: int = 18
    lower: int = 8
    release: int = 6
    retract: int = 8
    home: int = 14


def _resolve_phase(
    poses: list[np.ndarray],
    phase: Phase,
    cyl_idx: int,
    gripper: float,
    attached: bool,
    grasp_T_rel: np.ndarray | None,
    q_seed: np.ndarray,
) -> tuple[list[np.ndarray], list[WaypointInfo], np.ndarray]:
    """Resuelve IK para una fase y arma las metadatos."""
    Q_phase = []
    info_phase = []
    cur_seed = q_seed.copy()
    for T in poses:
        res = inverse_kinematics(T, q_init=cur_seed, n_restarts=3, tol_pos=1e-4)
        q_sol = res.q.copy()
        # Forzar el valor de gripper en q7 (no afecta a la pose del TCP)
        q_sol[6] = gripper
        Q_phase.append(q_sol)
        info_phase.append(WaypointInfo(
            phase=phase,
            cylinder_idx=cyl_idx,
            gripper=gripper,
            cylinder_attached=attached,
            T_target=T,
            grasp_T_rel=grasp_T_rel,
        ))
        cur_seed = q_sol  # warm-start
    return Q_phase, info_phase, cur_seed


def plan_pick_and_place(
    cylinders: list[Cylinder],
    place_offset: np.ndarray = np.array([0.020, 0.0, 0.0]),
    q_home: np.ndarray = Q_HOME_LAPARO,
    config: PhaseConfig | None = None,
    verbose: bool = False,
) -> tuple[PickPlacePlan, bool, str]:
    """Planea la trayectoria pick-and-place para todos los cilindros.

    Para cada cilindro `c_i` produce la secuencia:

        HOME -> approach(c_i) -> grasp(c_i) -> close ->
        lift(c_i) -> transit(c_i) -> lower(c_i_dest) ->
        release -> retract(c_i_dest)

    Y al final: -> home (regreso al HOME laparoscópico).

    Parameters
    ----------
    cylinders : list of Cylinder
        Cilindros a manipular.
    place_offset : np.ndarray, shape (3,)
        Desplazamiento aplicado a la posición original del cilindro para
        determinar el destino. Default: +2 cm en X mundo.
    q_home : np.ndarray, shape (7,)
        Configuración HOME del PSM.
    config : PhaseConfig, optional
        Número de waypoints por fase.

    Returns
    -------
    plan : PickPlacePlan
        Plan completo con articulaciones, metadata y poses finales.
    success : bool
        True si todos los waypoints fueron alcanzables (errores < 1e-3 m).
    message : str
        Resumen del resultado.
    """
    if config is None:
        config = PhaseConfig()

    T_home = forward_kinematics(q_home)
    Q_all: list[np.ndarray] = []
    info_all: list[WaypointInfo] = []
    phase_indices: dict[int, dict[Phase, tuple[int, int]]] = {}
    final_poses: list[np.ndarray] = [c.pose_homogeneous() for c in cylinders]

    cur_seed = q_home.copy()
    cur_T = T_home.copy()
    max_pos_err = 0.0
    max_rot_err = 0.0
    n_failed = 0

    for ci, cyl in enumerate(cylinders):
        phase_indices[ci] = {}

        # ---------- 1. APPROACH ----------
        T_grasp = compute_grasp_pose(cyl, T_home)
        T_approach = compute_approach_pose(T_grasp, offset=0.020)
        poses = interpolate_poses(cur_T, T_approach, config.approach)
        idx_start = len(Q_all)
        Qp, Ip, cur_seed = _resolve_phase(
            poses, "approach", ci, GRIPPER_OPEN_RAD, False, None, cur_seed,
        )
        Q_all.extend(Qp); info_all.extend(Ip)
        phase_indices[ci]["approach"] = (idx_start, len(Q_all) - 1)
        cur_T = T_approach.copy()

        # ---------- 2. GRASP ----------
        poses = interpolate_poses(cur_T, T_grasp, config.grasp)
        idx_start = len(Q_all)
        Qp, Ip, cur_seed = _resolve_phase(
            poses, "grasp", ci, GRIPPER_OPEN_RAD, False, None, cur_seed,
        )
        Q_all.extend(Qp); info_all.extend(Ip)
        phase_indices[ci]["grasp"] = (idx_start, len(Q_all) - 1)
        cur_T = T_grasp.copy()

        # Pose relativa cilindro->TCP en el momento del agarre.
        # Esta transformación se mantiene rígida durante lift+transit+lower.
        T_cyl_world = cyl.pose_homogeneous()
        T_tcp_world = T_grasp
        T_cyl_tcp = np.linalg.inv(T_tcp_world) @ T_cyl_world

        # ---------- 3. CLOSE (mordazas se cierran en sitio) ----------
        idx_start = len(Q_all)
        n_close = config.close
        for k in range(n_close):
            t = k / max(1, n_close - 1)
            grip = GRIPPER_OPEN_RAD * (1.0 - t) + GRIPPER_CLOSE_RAD * t
            res = inverse_kinematics(T_grasp, q_init=cur_seed)
            q = res.q.copy(); q[6] = grip
            Q_all.append(q)
            attached = (k == n_close - 1)  # solo el último wp tiene cilindro pegado
            info_all.append(WaypointInfo(
                phase="close", cylinder_idx=ci, gripper=grip,
                cylinder_attached=attached, T_target=T_grasp,
                grasp_T_rel=T_cyl_tcp if attached else None,
            ))
            cur_seed = q
        phase_indices[ci]["close"] = (idx_start, len(Q_all) - 1)

        # ---------- 4. LIFT ----------
        T_lift = compute_lift_pose(T_grasp, height=0.025)
        poses = interpolate_poses(T_grasp, T_lift, config.lift)
        idx_start = len(Q_all)
        Qp, Ip, cur_seed = _resolve_phase(
            poses, "lift", ci, GRIPPER_CLOSE_RAD, True, T_cyl_tcp, cur_seed,
        )
        Q_all.extend(Qp); info_all.extend(Ip)
        phase_indices[ci]["lift"] = (idx_start, len(Q_all) - 1)
        cur_T = T_lift.copy()

        # ---------- 5. TRANSIT ----------
        # Calcular destino: posición original + offset, vertical
        place_xy = cyl.position[:2] + place_offset[:2]
        T_place_grasp, place_pos = compute_place_grasp_pose(cyl, place_xy, T_home)
        T_place_lift = compute_lift_pose(T_place_grasp, height=0.025)

        poses = interpolate_poses(T_lift, T_place_lift, config.transit)
        idx_start = len(Q_all)
        Qp, Ip, cur_seed = _resolve_phase(
            poses, "transit", ci, GRIPPER_CLOSE_RAD, True, T_cyl_tcp, cur_seed,
        )
        Q_all.extend(Qp); info_all.extend(Ip)
        phase_indices[ci]["transit"] = (idx_start, len(Q_all) - 1)
        cur_T = T_place_lift.copy()

        # ---------- 6. LOWER ----------
        poses = interpolate_poses(T_place_lift, T_place_grasp, config.lower)
        idx_start = len(Q_all)
        Qp, Ip, cur_seed = _resolve_phase(
            poses, "lower", ci, GRIPPER_CLOSE_RAD, True, T_cyl_tcp, cur_seed,
        )
        Q_all.extend(Qp); info_all.extend(Ip)
        phase_indices[ci]["lower"] = (idx_start, len(Q_all) - 1)
        cur_T = T_place_grasp.copy()

        # ---------- 7. RELEASE (apertura de mordazas) ----------
        idx_start = len(Q_all)
        n_release = config.release
        for k in range(n_release):
            t = k / max(1, n_release - 1)
            grip = GRIPPER_CLOSE_RAD * (1.0 - t) + GRIPPER_OPEN_RAD * t
            res = inverse_kinematics(T_place_grasp, q_init=cur_seed)
            q = res.q.copy(); q[6] = grip
            Q_all.append(q)
            # El cilindro deja de estar pegado en el primer paso de release
            still_attached = (k == 0)
            info_all.append(WaypointInfo(
                phase="release", cylinder_idx=ci, gripper=grip,
                cylinder_attached=still_attached, T_target=T_place_grasp,
                grasp_T_rel=T_cyl_tcp if still_attached else None,
            ))
            cur_seed = q
        phase_indices[ci]["release"] = (idx_start, len(Q_all) - 1)

        # Actualizar pose final del cilindro: ahora está en place_pos vertical
        T_cyl_final = np.eye(4)
        T_cyl_final[:3, 3] = place_pos
        # cilindro vertical: rotación = identidad (eje z mundo)
        final_poses[ci] = T_cyl_final

        # ---------- 8. RETRACT (subir de nuevo, mordazas abiertas) ----------
        T_retract = compute_lift_pose(T_place_grasp, height=0.025)
        poses = interpolate_poses(T_place_grasp, T_retract, config.retract)
        idx_start = len(Q_all)
        Qp, Ip, cur_seed = _resolve_phase(
            poses, "retract", ci, GRIPPER_OPEN_RAD, False, None, cur_seed,
        )
        Q_all.extend(Qp); info_all.extend(Ip)
        phase_indices[ci]["retract"] = (idx_start, len(Q_all) - 1)
        cur_T = T_retract.copy()

    # ---------- HOME final ----------
    poses = interpolate_poses(cur_T, T_home, config.home)
    Qp, Ip, cur_seed = _resolve_phase(
        poses, "home", -1, GRIPPER_OPEN_RAD, False, None, cur_seed,
    )
    Q_all.extend(Qp); info_all.extend(Ip)

    # Verificar errores acumulados
    Q_arr = np.array(Q_all)
    for i, info in enumerate(info_all):
        T_ach = forward_kinematics(Q_arr[i])
        e_pos = np.linalg.norm(T_ach[:3, 3] - info.T_target[:3, 3])
        max_pos_err = max(max_pos_err, e_pos)
        if e_pos > 1.5e-2:  # Tolerancia de 15 mm para transiciones cartesianas de 5 GDL
            n_failed += 1

    success = (n_failed == 0)
    message = (
        f"Plan generado: {len(Q_all)} waypoints, "
        f"{len(cylinders)} cilindros, max_pos_err={max_pos_err*1000:.3f} mm, "
        f"waypoints fuera de tolerancia: {n_failed}"
    )
    if verbose:
        print(message)

    plan = PickPlacePlan(
        Q=Q_arr,
        waypoints=info_all,
        final_cylinder_poses=final_poses,
        phase_indices=phase_indices,
    )
    return plan, success, message


def replan_with_retries(
    rng_seed_initial: int = 0,
    n_cyl: int = 3,
    max_retries: int = 20,
    place_offset: np.ndarray = np.array([0.020, 0.0, 0.0]),
    q_home: np.ndarray = Q_HOME_LAPARO,
) -> tuple[list[Cylinder] | None, PickPlacePlan | None, int, str]:
    """Genera escenas hasta encontrar una totalmente alcanzable.

    Returns
    -------
    cylinders, plan, n_attempts, message
    """
    from psm_scene import generate_random_cylinders

    for attempt in range(max_retries):
        rng = np.random.default_rng(rng_seed_initial + attempt)
        try:
            cyls = generate_random_cylinders(n_cyl, rng=rng)
        except RuntimeError:
            continue
        plan, success, msg = plan_pick_and_place(
            cyls, place_offset=place_offset, q_home=q_home,
        )
        if success:
            return cyls, plan, attempt + 1, msg
    return None, None, max_retries, "Falló tras todos los reintentos."


from scipy.interpolate import interp1d


def resample_plan_and_fr5(
    plan: PickPlacePlan,
    fr5_q_per_wp: np.ndarray,
    target_n_points: int = 2000,
) -> tuple[PickPlacePlan, np.ndarray]:
    """Interpola un plan PickPlacePlan y las articulaciones del FR5 a N puntos exactos.

    Proporciona interpolación continua y fluida (ej. 2000 puntos) para eliminar
    vacíos, saltos y lograr animación y exportación de alta fidelidad.
    """
    N_orig = len(plan.waypoints)
    if N_orig == 0 or N_orig == target_n_points:
        return plan, fr5_q_per_wp

    t_orig = np.linspace(0.0, 1.0, N_orig)
    t_dense = np.linspace(0.0, 1.0, target_n_points)

    # Interpolación lineal articular continua para PSM (N, 7) y FR5 (N, 6)
    f_psm = interp1d(t_orig, plan.Q, axis=0, kind="linear")
    f_fr5 = interp1d(t_orig, fr5_q_per_wp, axis=0, kind="linear")

    Q_dense = f_psm(t_dense)
    fr5_dense = f_fr5(t_dense)

    # Mapeo de metadatos de waypoints por vecino más cercano
    indices = np.clip(np.round(t_dense * (N_orig - 1)).astype(int), 0, N_orig - 1)
    waypoints_dense = []
    for i_dense, idx in enumerate(indices):
        wp = plan.waypoints[idx]
        wp_interp = WaypointInfo(
            phase=wp.phase,
            cylinder_idx=wp.cylinder_idx,
            gripper=float(Q_dense[i_dense, 6]),  # valor exacto de q7 interpolado
            cylinder_attached=wp.cylinder_attached,
            T_target=wp.T_target,
            grasp_T_rel=wp.grasp_T_rel,
        )
        waypoints_dense.append(wp_interp)

    # Re-mapear sub-índices de fases
    phase_indices_dense = {}
    for cyl_idx, phases in plan.phase_indices.items():
        phase_indices_dense[cyl_idx] = {}
        for phase_name, (s_idx, e_idx) in phases.items():
            s_dense = int(round((s_idx / max(1, N_orig - 1)) * (target_n_points - 1)))
            e_dense = int(round((e_idx / max(1, N_orig - 1)) * (target_n_points - 1)))
            phase_indices_dense[cyl_idx][phase_name] = (s_dense, e_dense)

    plan_dense = PickPlacePlan(
        Q=Q_dense,
        waypoints=waypoints_dense,
        final_cylinder_poses=plan.final_cylinder_poses,
        phase_indices=phase_indices_dense,
    )
    return plan_dense, fr5_dense


__all__ = [
    "Phase",
    "WaypointInfo",
    "PickPlacePlan",
    "PhaseConfig",
    "interpolate_poses",
    "plan_pick_and_place",
    "replan_with_retries",
    "resample_plan_and_fr5",
]
