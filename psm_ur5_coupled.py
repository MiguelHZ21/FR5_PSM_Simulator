#!/usr/bin/env python3
"""
psm_ur5_coupled.py
===================

Sistema acoplado FR5 + PSM da Vinci EndoWrist.

El FR5 sostiene un trocar laparoscópico. El shaft del PSM atraviesa el
trocar y entra en una mesa de trabajo donde hay 3 cilindros aleatorios.
El PSM ejecuta una tarea de peg transfer (8 fases por cilindro, agarre
con mordazas que abren/cierran). En cada waypoint del PSM se proyecta
hacia el FR5 la pose del trocar requerida y se resuelve la IK del FR5.

Modos de configuración inicial:
    1. HOME del FR5  -> el HOME del FR5 define dónde queda el fulcro.
    2. Fulcro deseado -> el usuario provee posición + dirección del shaft,
       el FR5 resuelve IK para llegar ahí.

Si la IK del FR5 falla en algún waypoint, ese waypoint se omite (el PSM
sigue su trayectoria internamente, pero el FR5 mantiene la última
configuración válida).

Salidas:
    - sistema_completo.gif      : FR5 + trocar + PSM + cilindros
    - psm_solo.gif              : zoom del PSM sobre la mesa
    - errores_ur5.png           : errores IK del FR5 vs waypoint
    - posiciones_articulares_ur5.txt : trayectoria articular del FR5
    - posiciones_articulares_psm.txt : trayectoria articular del PSM

Uso:
    python psm_ur5_coupled.py                    # defaults
    python psm_ur5_coupled.py --mode home --seed 42
    python psm_ur5_coupled.py --mode fulcrum --fulcrum 0.55 0.0 0.4
    python psm_ur5_coupled.py --dist-trocar 0.20 --no-gif
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from io import BytesIO

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from psm_kinematics import (
    Q_HOME_LAPARO,
    forward_kinematics,
    forward_kinematics_full,
)
from psm_scene import (
    Cylinder,
    cylinder_mesh,
    generate_random_cylinders,
    TABLE_Z,
    TABLE_HALF_X,
    TABLE_HALF_Y_MIN,
    TABLE_HALF_Y_MAX,
)
from psm_pick_place import plan_pick_and_place, replan_with_retries, resample_plan_and_fr5
from psm_ur5_kinematics import (
    DEFAULT_DIST_TROCAR,
    DIST_TROCAR_RANGE,
    load_fr5,
    fr5_home_default,
    fr5_fk,
    fr5_ik,
    init_from_fr5_home,
    init_from_fulcrum,
    fr5_target_from_psm_state,
    transform_psm_pose_to_world,
)


# =============================================================================
# Constantes de visualización
# =============================================================================

COLOR_FR5_LINK   = "#1f6feb"
COLOR_FR5_JOINT  = "#37474F"
COLOR_TROCAR     = "#cc7700"
COLOR_FULCRUM    = "#ff7f0e"
COLOR_SHAFT      = "#3a3a3a"
COLOR_WRIST      = "#1f6feb"
COLOR_TCP        = "#d62728"
COLOR_TABLE      = "#e0e0d8"
COLOR_TABLE_EDGE = "#7a7a70"
AXIS_COLORS      = ("#e41a1c", "#4daf4a", "#377eb8")



# =============================================================================
# Geometría auxiliar para visualización
# =============================================================================

def _draw_cylinder_segment(ax, p1, p2, r, color, alpha=0.85, n=12):
    """Dibuja un cilindro 3D entre dos puntos (segmento del FR5 o del trocar)."""
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 1e-6:
        return
    z = v / L
    aux = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, aux); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circ = r * (np.outer(np.cos(th), x) + np.outer(np.sin(th), y))
    bot, top = p1 + circ, p2 + circ
    faces = [[bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]] for i in range(n)]
    poly = Poly3DCollection(faces, alpha=alpha, linewidth=0)
    poly.set_facecolor(color)
    ax.add_collection3d(poly)


def _draw_sphere(ax, c, r, color, alpha=0.9):
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 7)
    ax.plot_surface(
        c[0] + r * np.outer(np.cos(u), np.sin(v)),
        c[1] + r * np.outer(np.sin(u), np.sin(v)),
        c[2] + r * np.outer(np.ones(10), np.cos(v)),
        color=color, alpha=alpha, linewidth=0,
    )


def get_fr5_link_positions(fr5_robot, q_fr5):
    positions = [np.zeros(3)]
    T = np.eye(4)
    
    shoulder_offset = 0.135
    elbow_offset = -0.120

    T1 = fr5_robot.links[0].A(q_fr5[0])
    if not isinstance(T1, np.ndarray): T1 = T1.A
    T = T @ T1
    pos_shoulder = T[:3, 3].copy()
    positions.append(pos_shoulder)
    
    pos_shoulder_out = pos_shoulder + T[:3, 2] * shoulder_offset
    positions.append(pos_shoulder_out)
    
    T2 = fr5_robot.links[1].A(q_fr5[1])
    if not isinstance(T2, np.ndarray): T2 = T2.A
    T = T @ T2
    pos_elbow = T[:3, 3].copy() + T[:3, 2] * shoulder_offset
    positions.append(pos_elbow)
    
    pos_elbow_in = pos_elbow + T[:3, 2] * elbow_offset
    positions.append(pos_elbow_in)
    
    T3 = fr5_robot.links[2].A(q_fr5[2])
    if not isinstance(T3, np.ndarray): T3 = T3.A
    T = T @ T3
    pos_wrist1 = T[:3, 3].copy() + T[:3, 2] * (shoulder_offset + elbow_offset)
    positions.append(pos_wrist1)
    
    for i in range(3, 6):
        T_link = fr5_robot.links[i].A(q_fr5[i])
        if not isinstance(T_link, np.ndarray): T_link = T_link.A
        T = T @ T_link
        positions.append(T[:3, 3].copy())
        
    return positions


def draw_fr5(ax, fr5_robot, q_fr5):
    pos = get_fr5_link_positions(fr5_robot, q_fr5)
    n_links = len(pos) - 1
    radii_map = [0.045, 0.045, 0.040, 0.040, 0.038, 0.035, 0.030, 0.025]
    for i in range(n_links):
        r = radii_map[i] if i < len(radii_map) else 0.020
        _draw_cylinder_segment(ax, pos[i], pos[i + 1], r * 0.5, COLOR_FR5_LINK, alpha=0.85)
        if i in [0, 2, 4, 5, 6, 7]:
            _draw_sphere(ax, pos[i], r * 0.6, COLOR_FR5_JOINT)
    _draw_sphere(ax, pos[-1], 0.015, "#FF5722")


def draw_trocar(ax, T_fr5_tcp, dist_trocar):
    """Dibuja el trocar como un cilindro naranja entre el TCP FR5 y el fulcro."""
    p1 = T_fr5_tcp[:3, 3]
    p2 = p1 + dist_trocar * T_fr5_tcp[:3, 2]
    _draw_cylinder_segment(ax, p1, p2, 0.012, COLOR_TROCAR, alpha=0.85, n=14)


def draw_psm_in_world(ax, q_psm, T_psm_base_world, draw_triads=False):
    """Dibuja el PSM (shaft + EndoWrist + mordazas) en el frame mundo."""
    _, frames_psm = forward_kinematics_full(q_psm)
    # Transformar todos los frames al mundo
    origins_world = []
    for T in frames_psm:
        T_world = T_psm_base_world @ T
        origins_world.append(T_world[:3, 3])
    origins_world = np.array(origins_world)

    # Shaft (frame 0 a frame 4)
    ax.plot(origins_world[[0, 4], 0], origins_world[[0, 4], 1],
            origins_world[[0, 4], 2], color=COLOR_SHAFT, linewidth=4.0)
    # EndoWrist (frames 4-7)
    ax.plot(origins_world[4:, 0], origins_world[4:, 1], origins_world[4:, 2],
            color=COLOR_WRIST, linewidth=2.5)

    # Mordazas: dibujar en el TCP del PSM en el mundo
    T_tcp_world = T_psm_base_world @ frames_psm[7]
    R_tcp = T_tcp_world[:3, :3]
    p_tcp = T_tcp_world[:3, 3]
    finger_len = 0.012
    half_open = 0.5 * float(q_psm[6])
    for sign in (+1, -1):
        ang = sign * half_open
        finger_local = np.array([np.sin(ang) * finger_len, 0.0, np.cos(ang) * finger_len])
        finger_world = p_tcp + R_tcp @ finger_local
        ax.plot([p_tcp[0], finger_world[0]],
                [p_tcp[1], finger_world[1]],
                [p_tcp[2], finger_world[2]],
                color=COLOR_TCP, linewidth=2.5)

    ax.scatter(*origins_world[7], color=COLOR_TCP, s=45, marker="s",
               edgecolors="black", zorder=5)
    if draw_triads:
        for i, color in enumerate(AXIS_COLORS):
            o = T_tcp_world[:3, 3]
            e = o + 0.012 * T_tcp_world[:3, i]
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]],
                    color=color, linewidth=1.4)


def draw_table_in_world(ax, T_psm_base_world):
    """Dibuja la mesa de trabajo (en el frame del PSM, transformada al mundo)."""
    table_x = np.array([-TABLE_HALF_X, TABLE_HALF_X, TABLE_HALF_X, -TABLE_HALF_X])
    table_y = np.array([TABLE_HALF_Y_MIN, TABLE_HALF_Y_MIN,
                        TABLE_HALF_Y_MAX, TABLE_HALF_Y_MAX])
    table_z = np.full_like(table_x, TABLE_Z)
    pts_local = np.vstack([table_x, table_y, table_z])  # (3, 4)
    pts_world = T_psm_base_world[:3, :3] @ pts_local + T_psm_base_world[:3, 3:4]
    verts = [list(zip(pts_world[0], pts_world[1], pts_world[2]))]
    poly = Poly3DCollection(verts, alpha=0.30, facecolor=COLOR_TABLE,
                            edgecolor=COLOR_TABLE_EDGE)
    ax.add_collection3d(poly)


def draw_cylinder_in_world(ax, cyl_local, T_psm_base_world,
                            T_cyl_local: np.ndarray | None = None,
                            attached: bool = False):
    """Dibuja un cilindro de la escena en el frame mundo.

    Si T_cyl_local es None, usa la pose por defecto del cilindro.
    """
    if T_cyl_local is None:
        T_cyl_local = cyl_local.pose_homogeneous()
    T_cyl_world = T_psm_base_world @ T_cyl_local
    cyl_world = Cylinder(
        position=T_cyl_world[:3, 3],
        rotation=T_cyl_world[:3, :3],
        radius=cyl_local.radius,
        height=cyl_local.height,
        color=cyl_local.color,
        label=cyl_local.label,
    )
    X, Y, Z = cylinder_mesh(cyl_world, n_circ=20)
    a = 0.85 if attached else 0.7
    ax.plot_surface(X, Y, Z, color=cyl_world.color, alpha=a,
                    edgecolor="black", linewidth=0.3, shade=True)


# =============================================================================
# Renderizadores: sistema completo y solo PSM
# =============================================================================

def render_full_system(
    fig, fr5_robot, q_fr5, T_fr5_tcp, dist_trocar, p_fulcrum,
    q_psm, T_psm_base_world, cylinders, cyl_poses_local, attached_idx,
    info_text=None,
):
    """Renderiza el sistema completo: FR5 + trocar + PSM + cilindros + mesa."""
    fig.clf()
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    # Vista enfocada en el FR5+fulcro (no demasiado alejada)
    ax.set_xlim(-0.10, 0.95)
    ax.set_ylim(-0.55, 0.50)
    ax.set_zlim(0.0, 1.00)
    ax.set_xlabel("X [m]", color="white", fontsize=8)
    ax.set_ylabel("Y [m]", color="white", fontsize=8)
    ax.set_zlabel("Z [m]", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.view_init(elev=22, azim=42)
    try:
        ax.set_box_aspect((1.05, 1.05, 1.0))
    except AttributeError:
        pass

    # 1. Suelo (referencia visual)
    floor_x = np.array([-0.1, 0.95, 0.95, -0.1])
    floor_y = np.array([-0.55, -0.55, 0.50, 0.50])
    floor_z = np.zeros_like(floor_x)
    floor_verts = [list(zip(floor_x, floor_y, floor_z))]
    floor_poly = Poly3DCollection(floor_verts, alpha=0.10,
                                    facecolor="#444466", edgecolor="none")
    ax.add_collection3d(floor_poly)

    # 2. FR5
    draw_fr5(ax, fr5_robot, q_fr5)

    # 3. Trocar
    draw_trocar(ax, T_fr5_tcp, dist_trocar)

    # 4. Fulcro
    ax.scatter(*p_fulcrum, color=COLOR_FULCRUM, s=200, marker="*",
               edgecolors="black", zorder=10)

    # 5. Mesa
    draw_table_in_world(ax, T_psm_base_world)

    # 6. Cilindros
    for i, c in enumerate(cylinders):
        draw_cylinder_in_world(ax, c, T_psm_base_world,
                                T_cyl_local=cyl_poses_local[i],
                                attached=(i == attached_idx))

    # 7. PSM
    draw_psm_in_world(ax, q_psm, T_psm_base_world)

    # 8. Texto informativo
    if info_text:
        ax.text2D(0.02, 0.97, info_text, transform=ax.transAxes,
                  fontsize=9, verticalalignment="top", color="white",
                  family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#2d2d2d",
                            alpha=0.85, edgecolor="#888"))


def render_psm_only(
    fig, q_psm, T_psm_base_world, p_fulcrum,
    cylinders, cyl_poses_local, attached_idx, info_text=None,
):
    """Renderiza solo el PSM con la mesa y los cilindros (zoom)."""
    fig.clf()
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Calcular bounding box: mesa + fulcro + zona PSM
    # Tomar todos los puntos relevantes en el mundo y calcular un bounding box
    table_x = np.array([-TABLE_HALF_X, TABLE_HALF_X, TABLE_HALF_X, -TABLE_HALF_X])
    table_y = np.array([TABLE_HALF_Y_MIN, TABLE_HALF_Y_MIN,
                        TABLE_HALF_Y_MAX, TABLE_HALF_Y_MAX])
    table_z = np.full_like(table_x, TABLE_Z)
    pts_local = np.vstack([table_x, table_y, table_z])
    pts_table_world = (T_psm_base_world[:3, :3] @ pts_local
                       + T_psm_base_world[:3, 3:4]).T  # (4, 3)

    pts_to_include = np.vstack([
        pts_table_world,
        p_fulcrum.reshape(1, 3),
    ])
    p_min = pts_to_include.min(axis=0) - 0.04
    p_max = pts_to_include.max(axis=0) + 0.04
    rng = (p_max - p_min).max()
    center = (p_min + p_max) / 2.0
    half = rng / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("X [m]", fontsize=8)
    ax.set_ylabel("Y [m]", fontsize=8)
    ax.set_zlabel("Z [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=42)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    # Mesa
    draw_table_in_world(ax, T_psm_base_world)

    # Cilindros
    for i, c in enumerate(cylinders):
        draw_cylinder_in_world(ax, c, T_psm_base_world,
                                T_cyl_local=cyl_poses_local[i],
                                attached=(i == attached_idx))
        # Etiqueta
        T_cyl_world = T_psm_base_world @ cyl_poses_local[i]
        label_pos = T_cyl_world[:3, 3] + np.array([0, 0, c.height / 2 + 0.005])
        ax.text(*label_pos, c.label, color="black", fontsize=8,
                ha="center", weight="bold")

    # Fulcro
    ax.scatter(*p_fulcrum, color=COLOR_FULCRUM, s=140, marker="*",
               edgecolors="black", zorder=10)

    # PSM
    draw_psm_in_world(ax, q_psm, T_psm_base_world, draw_triads=True)

    if info_text:
        ax.text2D(0.02, 0.97, info_text, transform=ax.transAxes,
                  fontsize=9, verticalalignment="top",
                  family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            alpha=0.85, edgecolor="#888"))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sistema acoplado FR5 + PSM con peg transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["home", "fulcrum"], default="home",
                   help="Modo de inicialización: 'home' usa el HOME del FR5; "
                        "'fulcrum' usa la posición/dirección de fulcro indicadas.")
    p.add_argument("--fulcrum", nargs=3, type=float, default=[0.55, -0.10, 0.40],
                   metavar=("X", "Y", "Z"),
                   help="Posición del fulcro en el mundo (modo 'fulcrum') [m].")
    p.add_argument("--shaft-axis", nargs=3, type=float,
                   default=[0.5, 0.0, -0.866],
                   metavar=("NX", "NY", "NZ"),
                   help="Dirección unitaria del shaft del PSM (se normaliza).")
    p.add_argument("--dist-trocar", type=float, default=DEFAULT_DIST_TROCAR,
                   help=f"Distancia TCP FR5 -> fulcro [m] "
                        f"(rango: {DIST_TROCAR_RANGE[0]}-{DIST_TROCAR_RANGE[1]} m).")
    p.add_argument("--n-cyl", type=int, default=3, help="Número de cilindros.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed aleatoria de la escena (None = aleatoria).")
    p.add_argument("--offset-x", type=float, default=0.020,
                   help="Desplazamiento X de colocación [m].")
    p.add_argument("--fps", type=int, default=12, help="FPS del GIF.")
    p.add_argument("--step", type=int, default=3,
                   help="Submuestreo de waypoints en GIFs.")
    p.add_argument("--no-gif", action="store_true",
                   help="No generar GIFs (solo plan, errores, txt).")
    p.add_argument("--output-dir", type=str, default="./resultados_ur5_psm",
                   help="Directorio de salida.")
    return p.parse_args()


# =============================================================================
# Pipeline principal
# =============================================================================

def section(title):
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def main() -> int:
    args = parse_args()

    # Validar dist_trocar
    lo, hi = DIST_TROCAR_RANGE
    if not (lo <= args.dist_trocar <= hi):
        print(f"ERROR: dist-trocar debe estar entre {lo} y {hi} m. "
              f"Recibido: {args.dist_trocar}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    section("Sistema acoplado FR5 + PSM da Vinci EndoWrist")
    print(f"Modo de inicialización: {args.mode}")
    print(f"Distancia TCP FR5 -> fulcro: {args.dist_trocar*100:.1f} cm")
    print(f"Cilindros: {args.n_cyl}")
    print(f"Salida: {os.path.abspath(args.output_dir)}")

    # -------------------------------------------------------------------------
    # 1. Inicializar FR5 y calcular fulcro
    # -------------------------------------------------------------------------
    section("1. Inicialización del FR5 y cálculo del fulcro")
    fr5_robot, fr5_T_tool = load_fr5()

    if args.mode == "home":
        q_fr5_init, T_fr5_init, p_fulcrum, T_psm_base = init_from_fr5_home(
            fr5_robot, fr5_T_tool, dist_trocar=args.dist_trocar,
        )
        print(f"q_FR5 HOME (deg): {np.rad2deg(q_fr5_init).round(2)}")
    else:
        p_fulc_req = np.array(args.fulcrum)
        n_req = np.array(args.shaft_axis)
        n_req = n_req / np.linalg.norm(n_req)
        print(f"Fulcro deseado: {p_fulc_req}")
        print(f"Dirección shaft deseada: {n_req}")
        q_fr5_init, T_fr5_init, p_fulcrum, T_psm_base, ok, err = init_from_fulcrum(
            fr5_robot, fr5_T_tool, p_fulc_req, n_req, dist_trocar=args.dist_trocar,
        )
        if not ok:
            print(f"ERROR: IK del FR5 no alcanza el fulcro deseado "
                  f"(err = {err:.2f} mm). Pruebe otra pose.")
            return 1
        print(f"q_FR5 (deg): {np.rad2deg(q_fr5_init).round(2)}")

    print(f"TCP FR5: {T_fr5_init[:3,3].round(4)}")
    print(f"Fulcro:  {p_fulcrum.round(4)}")
    print(f"Eje shaft (mundo): {T_fr5_init[:3,2].round(4)}")

    # -------------------------------------------------------------------------
    # 2. Generar escena aleatoria del PSM (en frame del PSM)
    # -------------------------------------------------------------------------
    section("2. Escena aleatoria del PSM")
    seed = args.seed if args.seed is not None else np.random.randint(1, 10_000_000)
    print(f"Seed: {seed}")

    # IMPORTANTE: el frame base del PSM se orienta en el mundo de modo que
    # el shaft del PSM en su HOME laparoscópico salga del fulcro alineado
    # con la dirección del trocar sostenido por el FR5. Por tanto, la tarea
    # del PSM se planifica internamente con su Q_HOME_LAPARO original (que
    # es la configuración que produce trayectorias alcanzables sobre la mesa).
    cyls, plan, attempts, msg = replan_with_retries(
        rng_seed_initial=seed, n_cyl=args.n_cyl,
        place_offset=np.array([args.offset_x, 0.0, 0.0]),
        q_home=Q_HOME_LAPARO,
        max_retries=20,
    )
    if plan is None:
        print(f"ERROR: no se generó escena alcanzable: {msg}")
        return 1
    print(f"Escena alcanzable en {attempts} intento(s).")
    print(msg)
    for c in cyls:
        kind = "vertical" if c.is_vertical else "horizontal"
        print(f"  {c.label} ({kind}): "
              f"({c.position[0]:+.4f}, {c.position[1]:+.4f}, "
              f"{c.position[2]:+.4f}) m  [frame PSM]")

    # -------------------------------------------------------------------------
    # 3. Para cada waypoint del PSM, calcular IK del FR5
    # -------------------------------------------------------------------------
    section("3. Proyección PSM -> FR5: IK del FR5 por waypoint")
    n_wp = len(plan.waypoints)
    print(f"Waypoints totales del PSM: {n_wp}")

    fr5_q_per_wp = np.zeros((n_wp, 6))
    fr5_err_per_wp = np.zeros(n_wp)
    fr5_success_per_wp = np.zeros(n_wp, dtype=bool)
    q_fr5_seed = q_fr5_init.copy()
    n_failed = 0

    for i in range(n_wp):
        T_target = fr5_target_from_psm_state(
            plan.Q[i], p_fulcrum, T_psm_base,
            dist_trocar=args.dist_trocar,
        )
        q_sol, ok, err_mm = fr5_ik(fr5_robot, fr5_T_tool, T_target, q_fr5_seed)
        if ok:
            fr5_q_per_wp[i] = q_sol
            fr5_err_per_wp[i] = err_mm
            fr5_success_per_wp[i] = True
            q_fr5_seed = q_sol  # warm-start
        else:
            # Si falla, conservar la última config válida
            fr5_q_per_wp[i] = q_fr5_seed
            fr5_err_per_wp[i] = err_mm
            fr5_success_per_wp[i] = False
            n_failed += 1

        if (i + 1) % 50 == 0 or i == n_wp - 1:
            print(f"  {i+1}/{n_wp} waypoints  "
                  f"(éxito acumulado: {fr5_success_per_wp[:i+1].sum()})")

    print()
    print(f"IK FR5: {fr5_success_per_wp.sum()}/{n_wp} exitosos "
          f"({100*fr5_success_per_wp.mean():.1f}%)")
    print(f"Waypoints omitidos: {n_failed}")
    if fr5_success_per_wp.any():
        print(f"Error FR5 [mm]: media = {fr5_err_per_wp[fr5_success_per_wp].mean():.4f}, "
              f"máx = {fr5_err_per_wp[fr5_success_per_wp].max():.4f}")

    # Re-muestreo a 2000 puntos para alta fidelidad y suavidad
    print("\nRe-muestreando trayectoria a 2000 puntos...")
    plan, fr5_q_per_wp = resample_plan_and_fr5(plan, fr5_q_per_wp, target_n_points=2000)
    n_wp = len(plan.waypoints)

    # -------------------------------------------------------------------------
    # 4. Exportar trayectorias articulares a TXT
    # -------------------------------------------------------------------------
    section(f"4. Exportación de trayectorias articulares ({n_wp} puntos)")

    # 4a. FR5
    txt_fr5 = os.path.join(args.output_dir, "posiciones_articulares_fr5.txt")
    with open(txt_fr5, "w") as f:
        for i, wp in enumerate(plan.waypoints):
            q_deg = np.rad2deg(fr5_q_per_wp[i])
            # Offset a q1 para coincidir con el robot real
            q_deg[0] += 180.0
            if q_deg[0] > 180.0:
                q_deg[0] -= 360.0
            f.write(
                f"{q_deg[0]:.4f}, {q_deg[1]:.4f}, {q_deg[2]:.4f}, "
                f"{q_deg[3]:.4f}, {q_deg[4]:.4f}, {q_deg[5]:.4f};\n"
            )
    print(f"  {txt_fr5}")

    # 4b. PSM
    txt_psm = os.path.join(args.output_dir, "posiciones_articulares_psm.txt")
    with open(txt_psm, "w") as f:
        for i, wp in enumerate(plan.waypoints):
            q = plan.Q[i]
            q4 = np.rad2deg(q[3])
            q5 = np.rad2deg(q[4])
            q7 = np.rad2deg(q[6])
            q8 = -q7
            
            # Exportar q4, q5, q7, q8
            f.write(f"{q4:.4f}, {q5:.4f}, {q7:.4f}, {q8:.4f};\n")
    print(f"  {txt_psm}")

    # -------------------------------------------------------------------------
    # 5. Gráfica de errores IK del FR5
    # -------------------------------------------------------------------------
    section("5. Gráfica de errores IK del FR5")
    fig_err, ax_err = plt.subplots(figsize=(11, 4.5), facecolor="#1a1a2e")
    ax_err.set_facecolor("#2d2d2d")
    ax_err.plot(np.arange(n_wp) + 1, fr5_err_per_wp, color="#4FC3F7", lw=1.4,
                label="error IK FR5")
    if n_failed > 0:
        idx_fail = np.where(~fr5_success_per_wp)[0]
        ax_err.scatter(idx_fail + 1, fr5_err_per_wp[idx_fail],
                       color="#ff5555", s=18, label=f"fallos ({n_failed})")
    ax_err.set_xlabel("Waypoint", color="white")
    ax_err.set_ylabel("Error IK FR5 [mm]", color="white")
    ax_err.set_title("Error de IK del FR5 acoplado al PSM",
                     color="white", fontsize=11)
    ax_err.tick_params(colors="white")
    ax_err.grid(True, alpha=0.3, color="gray")
    for sp in ax_err.spines.values():
        sp.set_edgecolor("#555")
    ax_err.legend(loc="upper right", facecolor="#2d2d2d",
                  labelcolor="white", edgecolor="gray")
    plt.tight_layout()
    fig_err.savefig(os.path.join(args.output_dir, "errores_fr5.png"),
                    dpi=130, bbox_inches="tight",
                    facecolor=fig_err.get_facecolor())
    plt.close(fig_err)
    print(f"  errores_fr5.png")

    # -------------------------------------------------------------------------
    # 6. Generar GIFs (sistema completo y solo PSM)
    # -------------------------------------------------------------------------
    if args.no_gif:
        print("\n--no-gif: omitiendo generación de animaciones.")
    else:
        section("6. Generación de animaciones (puede tardar varios minutos)")
        generate_dual_gifs(
            fr5_robot=fr5_robot,
            fr5_T_tool=fr5_T_tool,
            fr5_q_per_wp=fr5_q_per_wp,
            fr5_success_per_wp=fr5_success_per_wp,
            plan=plan,
            cylinders=cyls,
            T_psm_base=T_psm_base,
            p_fulcrum=p_fulcrum,
            dist_trocar=args.dist_trocar,
            output_dir=args.output_dir,
            fps=args.fps,
            step=args.step,
        )

    section("Sistema acoplado FR5+PSM finalizado")
    print(f"Resultados en: {os.path.abspath(args.output_dir)}")
    return 0


def generate_dual_gifs(
    fr5_robot, fr5_T_tool, fr5_q_per_wp, fr5_success_per_wp, plan, cylinders,
    T_psm_base, p_fulcrum, dist_trocar, output_dir, fps=12, step=3,
):
    """Genera los dos GIFs sincronizados."""
    n_wp = len(plan.waypoints)
    indices = list(range(0, n_wp, step))
    print(f"Frames del GIF: {len(indices)} (de {n_wp} waypoints, paso={step})")

    cyl_poses_local = [c.pose_homogeneous() for c in cylinders]
    attached = None

    fig_full = Figure(figsize=(10, 8), dpi=85, facecolor="#1a1a2e")
    fig_psm  = Figure(figsize=(8, 7), dpi=85)

    frames_full = []
    frames_psm  = []

    for k, i in enumerate(indices):
        wp = plan.waypoints[i]
        q_psm = plan.Q[i]
        q_fr5 = fr5_q_per_wp[i]

        # Update cylinder poses (only consider state up to current index)
        for j_idx in range(max(0, indices[k - 1] if k > 0 else 0),
                            i + 1):
            wp_j = plan.waypoints[j_idx]
            if wp_j.cylinder_attached and wp_j.grasp_T_rel is not None:
                T_tcp_local = forward_kinematics(plan.Q[j_idx])
                cyl_poses_local[wp_j.cylinder_idx] = T_tcp_local @ wp_j.grasp_T_rel
                attached = wp_j.cylinder_idx
            elif j_idx > 0:
                prev = plan.waypoints[j_idx - 1]
                if (prev.cylinder_attached and not wp_j.cylinder_attached
                        and wp_j.cylinder_idx == prev.cylinder_idx):
                    cyl_poses_local[prev.cylinder_idx] = plan.final_cylinder_poses[prev.cylinder_idx]
                    attached = None

        # Pose actual del FR5
        T_fr5_actual = fr5_fk(fr5_robot, fr5_T_tool, q_fr5)

        # Info text
        ik_status = "OK" if fr5_success_per_wp[i] else "FAIL"
        info = (f"WP {i+1}/{n_wp}  step={k+1}/{len(indices)}\n"
                f"Fase: {wp.phase}  |  IK FR5: {ik_status}\n"
                f"Cil agarrado: "
                f"{'C' + str(wp.cylinder_idx) if wp.cylinder_attached else '-'}")

        # 6a. Sistema completo
        render_full_system(
            fig_full, fr5_robot, q_fr5, T_fr5_actual, dist_trocar, p_fulcrum,
            q_psm, T_psm_base, cylinders, cyl_poses_local, attached,
            info_text=info,
        )
        buf = BytesIO()
        fig_full.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                         facecolor=fig_full.get_facecolor())
        buf.seek(0)
        frames_full.append(Image.open(buf).convert("RGB"))

        # 6b. PSM solo
        render_psm_only(
            fig_psm, q_psm, T_psm_base, p_fulcrum,
            cylinders, cyl_poses_local, attached, info_text=info,
        )
        buf = BytesIO()
        fig_psm.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        frames_psm.append(Image.open(buf).convert("RGB"))

        if (k + 1) % 20 == 0 or k == len(indices) - 1:
            print(f"  frame {k+1}/{len(indices)}")

    duration = max(1, int(1000 / fps))

    path_full = os.path.join(output_dir, "sistema_completo.gif")
    print(f"  Guardando {path_full}...")
    frames_full[0].save(
        path_full, save_all=True, append_images=frames_full[1:],
        duration=duration, loop=0,
    )
    print(f"  -> {os.path.getsize(path_full) / 1024:.1f} KB")

    path_psm = os.path.join(output_dir, "psm_solo.gif")
    print(f"  Guardando {path_psm}...")
    frames_psm[0].save(
        path_psm, save_all=True, append_images=frames_psm[1:],
        duration=duration, loop=0,
    )
    print(f"  -> {os.path.getsize(path_psm) / 1024:.1f} KB")


if __name__ == "__main__":
    sys.exit(main())
