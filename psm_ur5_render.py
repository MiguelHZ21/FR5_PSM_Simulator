"""
psm_ur5_render.py
==================

Renderizado 3D del sistema acoplado FR5+PSM. Funciones de dibujo
puras, sin dependencia de PyQt ni del backend de matplotlib. Usadas
tanto por `psm_ur5_coupled.py` (CLI) como por `psm_ur5_simulator_gui.py`
(GUI interactiva).

(Se conserva el nombre de archivo original por compatibilidad de imports.)
"""

from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from psm_kinematics import forward_kinematics, forward_kinematics_full
from psm_scene import (
    Cylinder,
    cylinder_mesh,
    TABLE_Z,
    TABLE_HALF_X,
    TABLE_HALF_Y_MIN,
    TABLE_HALF_Y_MAX,
)


# Paleta de colores
COLOR_FR5_LINK   = "#1f6feb"
COLOR_FR5_JOINT  = "#37474F"
COLOR_TROCAR     = "#cc7700"
COLOR_FULCRUM    = "#ff7f0e"
COLOR_SHAFT      = "#3a3a3a"
COLOR_WRIST      = "#1f6feb"
COLOR_TCP        = "#d62728"
COLOR_TCP_FAIL   = "#ff0000"
COLOR_TABLE      = "#e0e0d8"
COLOR_TABLE_EDGE = "#7a7a70"
AXIS_COLORS      = ("#e41a1c", "#4daf4a", "#377eb8")


# =============================================================================
# Primitivas geométricas
# =============================================================================

def draw_cylinder_segment(ax, p1, p2, r, color, alpha=0.85, n=12):
    """Dibuja un cilindro 3D entre dos puntos."""
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)
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


def draw_sphere(ax, c, r, color, alpha=0.9, n_u=10, n_v=7):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    ax.plot_surface(
        c[0] + r * np.outer(np.cos(u), np.sin(v)),
        c[1] + r * np.outer(np.sin(u), np.sin(v)),
        c[2] + r * np.outer(np.ones(n_u), np.cos(v)),
        color=color, alpha=alpha, linewidth=0,
    )


# =============================================================================
# Componentes del sistema
# =============================================================================

def get_fr5_link_positions(fr5_robot, q_fr5):
    """Posiciones 3D de cada articulación del FR5 con offsets visuales."""
    from psm_ur5_kinematics import _FR5
    # pyrefly: ignore [missing-import]
    from fr5_kinematics import _dh
    
    positions = [np.zeros(3)]
    T = np.eye(4)
    
    # Offsets visuales similares al UR5 para que no se vea plano
    shoulder_offset = 0.135
    elbow_offset = -0.120

    # 1. Base a Hombro
    T1 = _dh(q_fr5[0], _FR5.d1, 0.0, np.pi/2)
    T = T @ T1
    pos_shoulder = T[:3, 3].copy()
    positions.append(pos_shoulder)
    
    # 2. Offset del hombro (cilindro horizontal de la articulación)
    pos_shoulder_out = pos_shoulder + T[:3, 2] * shoulder_offset
    positions.append(pos_shoulder_out)
    
    # 3. Hombro a Codo (Upper arm)
    T2 = _dh(q_fr5[1], 0.0, _FR5.a2, 0.0)
    T = T @ T2
    pos_elbow = T[:3, 3].copy() + T[:3, 2] * shoulder_offset
    positions.append(pos_elbow)
    
    # 4. Offset del codo
    pos_elbow_in = pos_elbow + T[:3, 2] * elbow_offset
    positions.append(pos_elbow_in)
    
    # 5. Codo a Muñeca 1 (Forearm)
    T3 = _dh(q_fr5[2], 0.0, _FR5.a3, 0.0)
    T = T @ T3
    pos_wrist1 = T[:3, 3].copy() + T[:3, 2] * (shoulder_offset + elbow_offset)
    positions.append(pos_wrist1)
    
    # 6, 7, 8: Muñeca 2, Muñeca 3, TCP
    T4 = _dh(q_fr5[3], _FR5.d4, 0.0, np.pi/2)
    T = T @ T4
    positions.append(T[:3, 3].copy())
    
    T5 = _dh(q_fr5[4], _FR5.d5, 0.0, -np.pi/2)
    T = T @ T5
    positions.append(T[:3, 3].copy())
    
    # Brida real de la articulación 6 del FR5 (d6_flange = 0.098 m)
    T_flange = T @ _dh(q_fr5[5], 0.098, 0.0, 0.0)
    positions.append(T_flange[:3, 3].copy())

    # TCP punta de la pinza (d6 = 0.457 m)
    T6 = T @ _dh(q_fr5[5], _FR5.d6, 0.0, 0.0)
    positions.append(T6[:3, 3].copy())
        
    return positions


def draw_fr5(ax, fr5_robot, q_fr5):
    """Dibuja el FR5 completo hasta su brida física (articulación 6)."""
    pos = get_fr5_link_positions(fr5_robot, q_fr5)
    # Dibuja la estructura azul del robot hasta la brida física (pos[-2])
    n_robot_links = len(pos) - 2
    radii_map = [0.045, 0.045, 0.040, 0.040, 0.038, 0.035, 0.030]
    for i in range(n_robot_links):
        r = radii_map[i] if i < len(radii_map) else 0.020
        draw_cylinder_segment(ax, pos[i], pos[i + 1], r * 0.5,
                              COLOR_FR5_LINK, alpha=0.85)
        if i in [0, 2, 4, 5, 6]:
            draw_sphere(ax, pos[i], r * 0.6, COLOR_FR5_JOINT)
    # Brida real del robot FR5
    draw_sphere(ax, pos[-2], 0.015, COLOR_FR5_JOINT)


def draw_trocar(ax, T_tcp, dist_trocar):
    """Dibuja el trocar (cilindro naranja alrededor del fulcro)."""
    p_tip = T_tcp[:3, 3]
    # Fulcro situado a dist_trocar hacia atrás (-Z) del TCP punta
    p_fulc = p_tip - dist_trocar * T_tcp[:3, 2]
    # Cilindro de trocar centrado en el fulcro (±5 cm)
    p1 = p_fulc - 0.05 * T_tcp[:3, 2]
    p2 = p_fulc + 0.05 * T_tcp[:3, 2]
    draw_cylinder_segment(ax, p1, p2, 0.012, COLOR_TROCAR, alpha=0.85, n=14)


def draw_psm_in_world(ax, q_psm, T_psm_base_world, ik_failed=False,
                      draw_triads=False, T_tcp=None):
    """Dibuja EndoWrist + vástago + mordazas del PSM."""
    from psm_kinematics import dh_modified_transform, L_PITCH2YAW, L_YAW2CTRL
    
    if T_tcp is not None:
        # El vástago va desde la brida del FR5 hasta la muñeca del EndoWrist
        # Posición de la brida real del FR5 (a -0.359m del TCP punta)
        p_flange = T_tcp[:3, 3] - (0.457 - 0.098) * T_tcp[:3, 2]
        
        q4, q5, q6, q7 = q_psm[3], q_psm[4], q_psm[5], q_psm[6]
        
        T4 = T_tcp @ dh_modified_transform(0.0, 0.0, 0.0, q4)
        T5 = T4 @ dh_modified_transform(-np.pi/2, 0.0, 0.0, -np.pi/2 + q5)
        T6 = T5 @ dh_modified_transform(-np.pi/2, L_PITCH2YAW, 0.0, -np.pi/2 + q6)
        T7 = T6 @ dh_modified_transform(-np.pi/2, 0.0, L_YAW2CTRL, q7)
        
        origins = np.array([p_flange, T4[:3, 3], T5[:3, 3], T6[:3, 3], T7[:3, 3]])
        
        # Shaft (Brida FR5 -> entrada de la muñeca EndoWrist)
        ax.plot([origins[0, 0], origins[1, 0]],
                [origins[0, 1], origins[1, 1]],
                [origins[0, 2], origins[1, 2]], color=COLOR_SHAFT, linewidth=4.0)
        
        # EndoWrist (links cortos)
        ax.plot(origins[1:, 0], origins[1:, 1], origins[1:, 2],
                color=COLOR_WRIST, linewidth=2.5)
        
        T_tip = T7
        
    else:
        # Fallback (para el panel derecho que dibuja solo el PSM)
        from psm_kinematics import forward_kinematics_full
        _, frames_psm = forward_kinematics_full(q_psm)
        origins = np.array([(T_psm_base_world @ T)[:3, 3] for T in frames_psm])
        ax.plot(origins[[0, 4], 0], origins[[0, 4], 1], origins[[0, 4], 2],
                color=COLOR_SHAFT, linewidth=4.0)
        ax.plot(origins[4:, 0], origins[4:, 1], origins[4:, 2],
                color=COLOR_WRIST, linewidth=2.5)
        T_tip = T_psm_base_world @ frames_psm[7]

    # Mordazas
    R_tcp = T_tip[:3, :3]
    p_tcp = T_tip[:3, 3]
    finger_len = 0.012
    half_open = 0.5 * abs(float(q_psm[6]))
    tcp_color = COLOR_TCP_FAIL if ik_failed else COLOR_TCP
    for sign in (+1, -1):
        ang = sign * half_open
        f_local = np.array([np.sin(ang) * finger_len, 0.0,
                            np.cos(ang) * finger_len])
        f_world = p_tcp + R_tcp @ f_local
        ax.plot([p_tcp[0], f_world[0]],
                [p_tcp[1], f_world[1]],
                [p_tcp[2], f_world[2]],
                color=tcp_color, linewidth=2.5)

    ax.scatter(*p_tcp, color=tcp_color, s=45, marker="s",
               edgecolors="black", zorder=5)
    
    if draw_triads:
        for i, color in enumerate(AXIS_COLORS):
            o = T_tip[:3, 3]
            e = o + 0.012 * T_tip[:3, i]
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]],
                    color=color, linewidth=1.4)


def draw_table_in_world(ax, T_psm_base_world):
    """Mesa de trabajo del simulador, transformada al mundo."""
    table_x = np.array([-TABLE_HALF_X, TABLE_HALF_X, TABLE_HALF_X, -TABLE_HALF_X])
    table_y = np.array([TABLE_HALF_Y_MIN, TABLE_HALF_Y_MIN,
                        TABLE_HALF_Y_MAX, TABLE_HALF_Y_MAX])
    table_z = np.full_like(table_x, TABLE_Z)
    pts_local = np.vstack([table_x, table_y, table_z])
    pts_world = (T_psm_base_world[:3, :3] @ pts_local
                 + T_psm_base_world[:3, 3:4])
    # Mesa totalmente horizontal en el mundo (sin ángulo de caída)
    pts_world[2, :] = np.mean(pts_world[2, :])
    verts = [list(zip(pts_world[0], pts_world[1], pts_world[2]))]
    poly = Poly3DCollection(verts, alpha=0.30, facecolor=COLOR_TABLE,
                            edgecolor=COLOR_TABLE_EDGE)
    ax.add_collection3d(poly)


def draw_cylinder_in_world(ax, cyl_local, T_psm_base_world,
                           T_cyl_local=None, attached=False):
    """Dibuja un cilindro de la escena en el frame mundo."""
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
# Renderizadores de alto nivel
# =============================================================================

def render_full_system(
    ax, fr5_robot, q_fr5, T_tcp, dist_trocar, p_fulcrum,
    q_psm, T_psm_base_world, cylinders, cyl_poses_local, attached_idx,
    ik_failed=False,
    info_text=None,
    xlim=(-0.80, 0.0), ylim=(-0.55, 0.50), zlim=(0.0, 1.00),
    elev=22, azim=-138,
    show_floor=True, info_color="white", bg_dark=True,
):
    """Renderiza el sistema completo en un Axes3D existente."""
    ax.clear()
    if bg_dark:
        ax.set_facecolor("#1a1a2e")
        tick_color = "white"
    else:
        ax.set_facecolor("white")
        tick_color = "black"

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X [m]", color=tick_color, fontsize=8)
    ax.set_ylabel("Y [m]", color=tick_color, fontsize=8)
    ax.set_zlabel("Z [m]", color=tick_color, fontsize=8)
    ax.tick_params(colors=tick_color, labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.05, 1.05, 1.0))
    except AttributeError:
        pass

    if show_floor:
        floor_x = np.array([xlim[0], xlim[1], xlim[1], xlim[0]])
        floor_y = np.array([ylim[0], ylim[0], ylim[1], ylim[1]])
        floor_z = np.zeros_like(floor_x)
        verts = [list(zip(floor_x, floor_y, floor_z))]
        floor = Poly3DCollection(
            verts,
            alpha=0.10 if bg_dark else 0.15,
            facecolor=("#444466" if bg_dark else "#aaaacc"),
            edgecolor="none",
        )
        ax.add_collection3d(floor)

    draw_fr5(ax, fr5_robot, q_fr5)
    draw_trocar(ax, T_tcp, dist_trocar)
    ax.scatter(*p_fulcrum, color=COLOR_FULCRUM, s=200, marker="*",
               edgecolors="black", zorder=10)
    draw_table_in_world(ax, T_psm_base_world)
    for i, c in enumerate(cylinders):
        draw_cylinder_in_world(ax, c, T_psm_base_world,
                               T_cyl_local=cyl_poses_local[i],
                               attached=(i == attached_idx))
    draw_psm_in_world(ax, q_psm, T_psm_base_world, ik_failed=ik_failed, T_tcp=T_tcp)

    if info_text:
        bbox_face = "#2d2d2d" if bg_dark else "white"
        ax.text2D(0.02, 0.97, info_text, transform=ax.transAxes,
                  fontsize=8, verticalalignment="top", color=info_color,
                  family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_face,
                            alpha=0.85, edgecolor="#888"))


def render_psm_only(
    ax, q_psm, T_psm_base_world, p_fulcrum,
    cylinders, cyl_poses_local, attached_idx,
    ik_failed=False,
    info_text=None, elev=22, azim=-138, zoom_factor=1.0,
):
    """Renderiza solo el PSM con la mesa y cilindros (auto-zoom)."""
    ax.clear()
    ax.set_facecolor("white")

    # Bounding box auto-calculado a partir de la mesa transformada
    table_x = np.array([-TABLE_HALF_X, TABLE_HALF_X, TABLE_HALF_X, -TABLE_HALF_X])
    table_y = np.array([TABLE_HALF_Y_MIN, TABLE_HALF_Y_MIN,
                        TABLE_HALF_Y_MAX, TABLE_HALF_Y_MAX])
    table_z = np.full_like(table_x, TABLE_Z)
    pts_local = np.vstack([table_x, table_y, table_z])
    pts_table_world = (T_psm_base_world[:3, :3] @ pts_local
                       + T_psm_base_world[:3, 3:4]).T
    pts_to_include = np.vstack([pts_table_world, p_fulcrum.reshape(1, 3)])
    p_min = pts_to_include.min(axis=0) - 0.04
    p_max = pts_to_include.max(axis=0) + 0.04
    rng = (p_max - p_min).max() / zoom_factor
    center = (p_min + p_max) / 2.0
    half = rng / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("X [m]", fontsize=8)
    ax.set_ylabel("Y [m]", fontsize=8)
    ax.set_zlabel("Z [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    draw_table_in_world(ax, T_psm_base_world)
    for i, c in enumerate(cylinders):
        draw_cylinder_in_world(ax, c, T_psm_base_world,
                               T_cyl_local=cyl_poses_local[i],
                               attached=(i == attached_idx))
        T_cyl_world = T_psm_base_world @ cyl_poses_local[i]
        label_pos = T_cyl_world[:3, 3] + np.array([0, 0, c.height / 2 + 0.005])
        ax.text(*label_pos, c.label, color="black", fontsize=8,
                ha="center", weight="bold")
    ax.scatter(*p_fulcrum, color=COLOR_FULCRUM, s=140, marker="*",
               edgecolors="black", zorder=10)
    draw_psm_in_world(ax, q_psm, T_psm_base_world, ik_failed=ik_failed,
                      draw_triads=True)

    if info_text:
        ax.text2D(0.02, 0.97, info_text, transform=ax.transAxes,
                  fontsize=8, verticalalignment="top", family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            alpha=0.85, edgecolor="#888"))


__all__ = [
    "draw_fr5", "draw_trocar", "draw_psm_in_world",
    "draw_table_in_world", "draw_cylinder_in_world",
    "render_full_system", "render_psm_only",
    "COLOR_FR5_LINK", "COLOR_TROCAR", "COLOR_FULCRUM",
]
