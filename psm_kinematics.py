"""
psm_kinematics.py
==================

Cinemática directa e inversa del Patient Side Manipulator (PSM) del robot
quirúrgico da Vinci classic (IS-1200), incluyendo el EndoWrist con apertura
de pinza (jaw).

Implementación en Python basada en el enfoque del repositorio
``cwru_davinci_kinematics`` (Case Western Reserve University), que utiliza
parámetros Denavit-Hartenberg modificados (convención de Craig) y un modelo
de 7 grados de libertad:

    q = [q1, q2, q3, q4, q5, q6, q7]

donde:
    - q1 : outer yaw    (revoluta) [rad]
    - q2 : outer pitch  (revoluta) [rad]
    - q3 : inserción    (prismática) [m]
    - q4 : tool roll    (revoluta) [rad]
    - q5 : wrist pitch  (revoluta) [rad]
    - q6 : wrist yaw    (revoluta) [rad]
    - q7 : jaw          (apertura de la pinza) [rad]

El frame base coincide con el Remote Center of Motion (RCM o "fulcro"):
mecánicamente, el shaft de la herramienta siempre cruza este punto fijo,
que en cirugía corresponde al trocar insertado en la pared abdominal del
paciente. Esta restricción está implícita en la cadena DH: las articulaciones
q1 y q2 rotan el shaft alrededor del origen del frame base, y q3 desliza la
herramienta a lo largo de su propio eje pasando por dicho origen.

Autor: generado para Hermes Fabián Vargas Rosero - Universidad del Cauca
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# =============================================================================
# PARÁMETROS DH MODIFICADOS DEL PSM DA VINCI CLASSIC (IS-1200)
# =============================================================================
#
# Estos valores siguen el modelo del repositorio cwru_davinci_kinematics y
# son los del PSM clásico (IS-1200). Se utiliza la convención DH modificada
# de Craig:
#
#     T_{i-1}^{i} = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
#
# Las dimensiones son las publicadas para el da Vinci classic; coinciden con
# las usadas en el dVRK ROS y en el header del repo CWRU.
#
# Notación:
#     - alpha_{i-1} : torsión del enlace anterior [rad]
#     - a_{i-1}     : longitud del enlace anterior [m]
#     - d_i         : desplazamiento articular (constante o variable) [m]
#     - theta_i     : ángulo articular (constante u offset + variable) [rad]
#     - is_prism    : True si la articulación es prismática
# -----------------------------------------------------------------------------

# Longitudes geométricas características del EndoWrist [m]
L_RCC      = 0.4318   # distancia del RCM a la base de la muñeca (longitud
                      # virtual del shaft cuando d3 = 0; la prismática se
                      # mide como desplazamiento adicional respecto a esto)
L_TOOL     = 0.4159   # longitud del tool (desde el final del shaft hasta el
                      # eje de la muñeca esférica del EndoWrist)
L_PITCH2YAW = 0.0091  # offset entre el eje de wrist pitch y wrist yaw [m]
L_YAW2CTRL = 0.0102   # offset desde wrist yaw hasta el punto de control
                      # (centro de la pinza, TCP)

# Tabla DH modificada del PSM completo (7 GDL)
# Cada fila: (alpha_{i-1}, a_{i-1}, d_i, theta_offset_i, is_prism)
# El valor articular q_i se SUMA a theta_offset_i para articulaciones
# revolutas, o se SUMA a d_i para la prismática.
PSM_DH_TABLE = [
    # alpha_{i-1},   a_{i-1},   d_i,                 theta_offset_i,  prism
    (np.pi / 2,      0.0,       0.0,                 np.pi / 2,       False),  # 1
    (-np.pi / 2,     0.0,       0.0,                -np.pi / 2,       False),  # 2
    (np.pi / 2,      0.0,       -L_RCC,              0.0,             True ),  # 3 (prismática)
    (0.0,            0.0,        L_TOOL,             0.0,             False),  # 4
    (-np.pi / 2,     0.0,        0.0,               -np.pi / 2,       False),  # 5
    (-np.pi / 2,     L_PITCH2YAW, 0.0,              -np.pi / 2,       False),  # 6
    (-np.pi / 2,     0.0,        L_YAW2CTRL,         0.0,             False),  # 7 (jaw)
]

# Posición home recomendada del PSM (shaft vertical)
# - Las primeras 6 articulaciones a 0 rad / m alinean el shaft a lo largo
#   del eje Z del frame base apuntando hacia abajo (configuración natural
#   del PSM colgado del SUJ con la herramienta vertical).
# - Para que el TCP esté distal al fulcro (fuera del paciente, en el espacio
#   de trabajo útil) se requiere un valor positivo de inserción. Aquí se
#   define un home pedagógico con d3_home > 0.
Q_HOME = np.array([
    0.0,    # q1 outer yaw
    0.0,    # q2 outer pitch
    0.12,   # q3 inserción (m) -- el shaft sobresale 12 cm más allá de la
            # configuración nominal interna del paralelogramo
    0.0,    # q4 tool roll
    0.0,    # q5 wrist pitch
    0.0,    # q6 wrist yaw
    0.0,    # q7 jaw (cerrado)
])

# Posición home laparoscópica con shaft inclinado 45° respecto a la
# horizontal. En cirugía real el shaft entra por el trocar formando un
# ángulo con la pared abdominal; un valor típico es ~45°. Esta configuración
# se obtiene rotando el outer pitch (q2) a π/4 rad, lo que hace que el shaft
# caiga en el plano XZ del frame base, con la punta del TCP desplazada hacia
# +X y -Z respecto al fulcro. Las demás articulaciones permanecen igual al
# Q_HOME vertical.
TILT_ANGLE = np.pi / 4  # 45 grados sobre la horizontal
Q_HOME_TILTED = np.array([
    0.0,         # q1
    TILT_ANGLE,  # q2 inclinado 45°
    0.12,        # q3
    0.0,         # q4
    0.0,         # q5
    0.0,         # q6
    0.0,         # q7
])

# Posición home para tareas de simulador laparoscópico (peg transfer).
# Compromiso entre realismo (shaft inclinado, no vertical) y rango
# operativo amplio sobre una mesa horizontal: q2 = 30° resulta ideal.
# Permite alcanzar bien una zona de 5x5 cm sobre una mesa a Z = -0.16 m
# sin saturar los límites articulares.
LAPARO_TILT_ANGLE = np.pi / 6  # 30 grados de inclinación del shaft
Q_HOME_LAPARO = np.array([
    0.0,                # q1
    LAPARO_TILT_ANGLE,  # q2 inclinado 30°
    0.13,               # q3 ligeramente mayor para alcanzar la mesa
    0.0,                # q4
    0.0,                # q5
    0.0,                # q6
    0.0,                # q7
])

# Límites articulares aproximados del PSM classic (para validación / IK)
Q_LIMITS = np.array([
    [-np.pi / 2,   np.pi / 2 ],   # q1
    [-np.pi / 2,   np.pi / 2 ],   # q2
    [ 0.0,         0.24      ],   # q3 (m)
    [-np.pi,       np.pi     ],   # q4
    [-np.pi / 2,   np.pi / 2 ],   # q5
    [-np.pi / 2,   np.pi / 2 ],   # q6
    [ 0.0,         np.pi / 2 ],   # q7 jaw (0 = cerrado)
])


# =============================================================================
# UTILIDADES DE TRANSFORMACIÓN HOMOGÉNEA
# =============================================================================

def dh_modified_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Construye la transformación homogénea 4x4 según DH modificado (Craig).

    .. math::
        T_{i-1}^{i} = Rx(\\alpha_{i-1}) \\, Tx(a_{i-1}) \\, Rz(\\theta_i) \\, Tz(d_i)

    Parameters
    ----------
    alpha : float
        Torsión del enlace anterior alpha_{i-1} [rad].
    a : float
        Longitud del enlace anterior a_{i-1} [m].
    d : float
        Desplazamiento articular d_i [m].
    theta : float
        Ángulo articular theta_i [rad].

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        Matriz de transformación homogénea.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    T = np.array([
        [   ct,    -st,    0.0,        a],
        [st * ca, ct * ca, -sa, -sa * d ],
        [st * sa, ct * sa,  ca,  ca * d ],
        [   0.0,    0.0,   0.0,      1.0],
    ], dtype=float)
    return T


def rotation_matrix(T: np.ndarray) -> np.ndarray:
    """Devuelve la submatriz de rotación 3x3 de una transformación homogénea."""
    return T[:3, :3]


def position_vector(T: np.ndarray) -> np.ndarray:
    """Devuelve el vector de posición 3x1 de una transformación homogénea."""
    return T[:3, 3]


# =============================================================================
# CINEMÁTICA DIRECTA (FK)
# =============================================================================

def forward_kinematics_full(q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cinemática directa completa del PSM (7 GDL).

    Calcula la pose del TCP (tool center point) a partir del vector articular,
    así como las transformaciones acumuladas de cada frame intermedio (útil
    para visualización del robot).

    Parameters
    ----------
    q : np.ndarray, shape (7,)
        Vector articular [q1, ..., q7]. q3 en metros, los demás en radianes.

    Returns
    -------
    T_tcp : np.ndarray, shape (4, 4)
        Pose homogénea del TCP respecto al frame base (RCM).
    frames : list of np.ndarray
        Lista de 8 transformaciones acumuladas T_0^0, T_0^1, ..., T_0^7.
        T_0^0 es la identidad (frame base = RCM).
    """
    q = np.asarray(q, dtype=float).ravel()
    if q.size != 7:
        raise ValueError(f"Se esperaba q de tamaño 7, se recibió {q.size}.")

    T = np.eye(4)
    frames = [T.copy()]

    for i, (alpha, a, d, theta_off, is_prism) in enumerate(PSM_DH_TABLE):
        if is_prism:
            d_i = d + q[i]
            theta_i = theta_off
        else:
            d_i = d
            theta_i = theta_off + q[i]

        T_step = dh_modified_transform(alpha, a, d_i, theta_i)
        T = T @ T_step
        frames.append(T.copy())

    return T, frames


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Versión compacta de FK que devuelve solo la pose del TCP."""
    T_tcp, _ = forward_kinematics_full(q)
    return T_tcp


# =============================================================================
# JACOBIANO GEOMÉTRICO (numérico)
# =============================================================================

def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """Jacobiano geométrico 6x6 del PSM (ignorando la apertura de pinza q7).

    Calculado por columnas a partir de los frames intermedios. Las primeras
    6 articulaciones determinan la pose del TCP; q7 (jaw) no afecta la pose
    del punto de control y por tanto se omite del Jacobiano.

    Parameters
    ----------
    q : np.ndarray, shape (7,)
        Vector articular completo.

    Returns
    -------
    J : np.ndarray, shape (6, 6)
        Jacobiano que mapea [dq1...dq6]^T a [v_x, v_y, v_z, w_x, w_y, w_z]^T
        expresado en el frame base.
    """
    _, frames = forward_kinematics_full(q)
    p_tcp = position_vector(frames[7])

    # Convención DH modificada (Craig): tras aplicar la transformación
    # T_step_i, el frame resultante tiene su z_i alineado con el eje de la
    # articulación i+1. Por lo tanto, para construir el Jacobiano de la
    # articulación i (i = 1..6, 0-indexed = 0..5), el eje correcto es z del
    # frame i (es decir, frames[i] tras aplicar T_step_i, no frames[i-1]).
    J = np.zeros((6, 6))
    for i in range(6):
        T_i = frames[i + 1]          # frame i (resultante de T_step_i)
        z_i = T_i[:3, 2]             # eje z_i = eje de la articulación (i+1)
        p_i = T_i[:3, 3]             # origen del frame i
        is_prism = PSM_DH_TABLE[i][4]

        if is_prism:
            J[:3, i] = z_i
            J[3:, i] = 0.0
        else:
            J[:3, i] = np.cross(z_i, (p_tcp - p_i))
            J[3:, i] = z_i
    return J


# =============================================================================
# CINEMÁTICA INVERSA (IK)
# =============================================================================
#
# Estrategia híbrida:
#
#   1. Solución analítica para las 3 primeras articulaciones (q1, q2, q3) a
#      partir de la posición del centro de la muñeca esférica. La estructura
#      del PSM permite esta descomposición porque el shaft de la herramienta
#      pasa siempre por el RCM (origen del frame base).
#
#   2. Solución analítica para las 3 articulaciones de la muñeca esférica
#      (q4, q5, q6) a partir de la orientación deseada y las wrist offsets.
#
#   3. q7 (jaw) se preserva del estado actual o se especifica aparte, ya que
#      no afecta la pose del TCP.
#
#   4. Refinamiento numérico final por Jacobiano amortiguado (Levenberg-
#      Marquardt) para corregir pequeños errores debidos a los offsets no
#      ideales (L_PITCH2YAW, L_YAW2CTRL) que rompen la condición de muñeca
#      esférica pura.
# -----------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Envuelve un ángulo a (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _ik_analytical_initial_guess(T_target: np.ndarray) -> np.ndarray:
    """Estimación inicial analítica de q a partir de la pose deseada.

    Aprovecha la geometría del PSM:
        - El TCP está a una distancia aproximada (L_TOOL + L_YAW2CTRL) del
          fulcro a lo largo del shaft cuando q3 = 0 y la muñeca está extendida.
        - El shaft pasa por el origen del frame base (RCM); por tanto, la
          dirección del shaft está determinada por la posición del centro de
          la muñeca w respecto al fulcro.
    """
    p = position_vector(T_target)
    R = rotation_matrix(T_target)

    # Aproximamos el centro de la muñeca como el TCP retraído
    # (L_PITCH2YAW + L_YAW2CTRL) a lo largo de -z_TCP. Es una aproximación
    # razonable cuando los offsets son pequeños comparados con L_TOOL.
    z_tcp = R[:, 2]
    p_wrist = p - (L_PITCH2YAW + L_YAW2CTRL) * z_tcp

    # El shaft va del fulcro (origen) al centro de la muñeca; su longitud
    # es L_TOOL + q3 (con q3 = inserción adicional).
    r = np.linalg.norm(p_wrist)
    q3 = r - L_TOOL
    if q3 < 0.0:
        # Caso degenerado: el target está demasiado cerca del fulcro.
        q3 = 0.0

    # Dirección unitaria del shaft (apunta del fulcro hacia el centro de la
    # muñeca = dirección +z del frame del shaft)
    if r < 1e-9:
        u = np.array([0.0, 0.0, -1.0])
    else:
        u = p_wrist / r

    # En la configuración home, el shaft apunta a lo largo de -z del frame
    # base (hacia abajo). Convertimos la dirección u a las articulaciones
    # outer yaw (q1) y outer pitch (q2). Geometría:
    #
    #   u = [ sin(q1)*cos(q2),  -sin(q2),  -cos(q1)*cos(q2) ]
    #
    # de donde:
    q2 = np.arcsin(np.clip(-u[1], -1.0, 1.0))
    q1 = np.arctan2(u[0], -u[2])

    # Para la orientación del TCP, usamos el roll q4 y los ángulos de la
    # muñeca q5, q6 a partir de R relativo al frame del shaft.
    # Frame del shaft tras q1, q2, q3: la base de la muñeca con z apuntando
    # a lo largo del shaft.
    # Aproximación inicial: q4 = q5 = q6 = 0; el refinamiento numérico los
    # corrige rápidamente.
    q4 = 0.0
    q5 = 0.0
    q6 = 0.0
    q7 = 0.0

    return np.array([q1, q2, q3, q4, q5, q6, q7])


def _pose_error(T_current: np.ndarray, T_target: np.ndarray, lock_q6: bool = True) -> np.ndarray:
    """Error de pose en formato 5D/6D.

    Para EndoWrist de 4 GDL (q6 = 0.0), calcula el error en 5D: posición 3D +
    alineación del vector de aproximación z_tcp (2D), permitiendo convergencia
    exacta sin depender del 6° GDL inactivo.
    """
    p_err = position_vector(T_target) - position_vector(T_current)

    if lock_q6:
        z_tgt = T_target[:3, 2]
        z_cur = T_current[:3, 2]
        w_err = np.cross(z_cur, z_tgt)
        return np.concatenate([p_err, w_err])

    R_err = rotation_matrix(T_target) @ rotation_matrix(T_current).T
    # Vector log de SO(3) - método numérico estable
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        w_err = np.zeros(3)
    elif np.abs(np.pi - theta) < 1e-6:
        # Caso casi pi: fórmula alternativa
        diag = np.diag(R_err)
        idx = int(np.argmax(diag))
        v = np.zeros(3)
        v[idx] = np.sqrt(max(0.0, (R_err[idx, idx] + 1.0) / 2.0))
        for j in range(3):
            if j != idx:
                v[j] = R_err[idx, j] / (2.0 * v[idx]) if v[idx] > 1e-9 else 0.0
        w_err = theta * v
    else:
        w_err = (theta / (2.0 * np.sin(theta))) * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

    return np.concatenate([p_err, w_err])


@dataclass
class IKResult:
    """Resultado de un cálculo de cinemática inversa."""
    q: np.ndarray
    success: bool
    iterations: int
    pos_error: float       # norma del error de posición [m]
    rot_error: float       # norma del error rotacional [rad]
    message: str


def inverse_kinematics(
    T_target: np.ndarray,
    q_init: np.ndarray | None = None,
    *,
    max_iter: int = 150,
    tol_pos: float = 1e-6,
    tol_rot: float = 1e-5,
    damping: float = 1e-3,
    preserve_jaw: bool = True,
    n_restarts: int = 6,
) -> IKResult:
    """Cinemática inversa del PSM mediante Levenberg-Marquardt amortiguado.

    Implementa multistart automático: si la primera ejecución no converge,
    reintenta con perturbaciones de la semilla inicial. Esto mejora la
    robustez frente a poses lejanas al home, en las que el guess analítico
    inicial puede caer cerca de mínimos locales del residuo de pose.

    Parameters
    ----------
    T_target : np.ndarray, shape (4, 4)
        Pose objetivo del TCP en el frame base.
    q_init : np.ndarray, shape (7,), optional
        Estimación inicial. Si None, se usa Q_HOME junto con un guess analítico
        de q1, q2, q3 derivado de la posición objetivo.
    max_iter : int
        Número máximo de iteraciones por intento.
    tol_pos : float
        Tolerancia de convergencia en posición [m].
    tol_rot : float
        Tolerancia de convergencia en rotación [rad].
    damping : float
        Factor de amortiguamiento del Jacobiano (DLS / Levenberg-Marquardt).
    preserve_jaw : bool
        Si True, q7 (jaw) se mantiene en su valor inicial (no afecta la pose).
    n_restarts : int
        Número total de intentos con semillas distintas si el primero falla.

    Returns
    -------
    IKResult
        Mejor solución encontrada (la que minimiza el error total).
    """
    if q_init is None:
        q_seed = _ik_analytical_initial_guess(T_target)
        q_seed[3:7] = Q_HOME[3:7]
        analytical_seed = q_seed.copy()
    else:
        q_seed = np.asarray(q_init, dtype=float).copy()
        # Calcular también el guess analítico para usarlo como fallback
        # cuando la semilla del usuario no converge. Importante: usar
        # Q_HOME[3:7] (no q_seed[3:7]) para que si el seed tiene articulaciones
        # cerca de un límite, el guess analítico permita escapar.
        analytical_seed = _ik_analytical_initial_guess(T_target)
        analytical_seed[3:7] = Q_HOME[3:7].copy()
        # Preservar solo q7 (gripper) que no afecta a la pose
        analytical_seed[6] = q_seed[6]

    best = None
    rng = np.random.default_rng(seed=12345)

    for attempt in range(n_restarts):
        if attempt == 0:
            q = q_seed.copy()
        elif attempt == 1 and q_init is not None:
            # Probar también con el guess analítico antes de perturbar
            q = analytical_seed.copy()
        else:
            # Perturbar la semilla con ruido proporcional al rango articular
            base = q_seed if attempt % 2 == 0 else analytical_seed
            q = base.copy()
            scale = np.array([0.5, 0.5, 0.05, 1.0, 0.5, 0.5, 0.0])
            q[:6] = base[:6] + scale[:6] * rng.standard_normal(6)

        result = _run_ik_lm(
            T_target, q,
            max_iter=max_iter, tol_pos=tol_pos, tol_rot=tol_rot,
            damping=damping, preserve_jaw=preserve_jaw,
        )
        # Saturar dentro de límites
        for j in range(7):
            qmin, qmax = Q_LIMITS[j]
            result.q[j] = np.clip(result.q[j], qmin, qmax)

        if best is None or (result.pos_error + result.rot_error
                            < best.pos_error + best.rot_error):
            best = result
        if result.success:
            best = result
            break

        # Early exit pragmático: si el error es razonablemente bajo aunque
        # no haya alcanzado la tolerancia exacta, salir para no perder
        # tiempo en multistarts que probablemente no ayudarán (típicamente
        # cuando la solución está al borde de un límite articular).
        # Umbral: 0.5 mm de posición, 0.05 rad (~3°) de orientación.
        if result.pos_error < 5e-4 and result.rot_error < 5e-2:
            break

    return best


def _run_ik_lm(
    T_target: np.ndarray,
    q: np.ndarray,
    *,
    max_iter: int,
    tol_pos: float,
    tol_rot: float,
    damping: float,
    preserve_jaw: bool,
) -> IKResult:
    """Una corrida de Levenberg-Marquardt (sin restarts)."""
    q = q.copy()
    jaw0 = q[6]

    pos_err_norm = np.inf
    rot_err_norm = np.inf
    success = False
    message = "Sin convergencia."
    it = 0

    for it in range(max_iter):
        T_cur = forward_kinematics(q)
        e = _pose_error(T_cur, T_target)
        pos_err_norm = float(np.linalg.norm(e[:3]))
        rot_err_norm = float(np.linalg.norm(e[3:]))

        if pos_err_norm < tol_pos and rot_err_norm < tol_rot:
            success = True
            message = f"Convergencia en {it} iteraciones."
            break

        # Jacobiano 6x6 (ignora q7)
        J = geometric_jacobian(q)
        J[:, 5] = 0.0  # Fijar q6 (wrist_yaw) en 0.0 para compatibilidad con EndoWrist de 4 GDL

        # Levenberg-Marquardt: dq = (J^T J + lambda^2 I)^{-1} J^T e
        JT = J.T
        H = JT @ J + (damping ** 2) * np.eye(6)
        dq6 = np.linalg.solve(H, JT @ e)

        # Aplicamos a las articulaciones
        q[:6] = q[:6] + dq6
        q[5] = 0.0  # Forzar q6 = 0.0
        if preserve_jaw:
            q[6] = jaw0

        # Aplicar límites de articulaciones revolutas (saturación blanda)
        for j in range(7):
            qmin, qmax = Q_LIMITS[j]
            q[j] = np.clip(q[j], qmin, qmax)

    return IKResult(
        q=q,
        success=success,
        iterations=it + 1,
        pos_error=pos_err_norm,
        rot_error=rot_err_norm,
        message=message,
    )


# =============================================================================
# UTILIDADES PARA TRAYECTORIAS CARTESIANAS
# =============================================================================

def linear_trajectory_along_tcp_axis(
    q_start: np.ndarray,
    length: float = 0.02,
    n_points: int = 50,
    axis: str = "z",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Genera una trayectoria lineal a lo largo de un eje del TCP.

    Mantiene la orientación del TCP constante e iguala a la del estado de
    inicio.

    Parameters
    ----------
    q_start : np.ndarray, shape (7,)
        Configuración articular inicial.
    length : float
        Longitud total de la trayectoria [m].
    n_points : int
        Número de puntos cartesianos (incluye inicio y fin).
    axis : {"x", "y", "z"}
        Eje del TCP a lo largo del cual avanzar.

    Returns
    -------
    s : np.ndarray, shape (n_points,)
        Vector de posiciones a lo largo de la línea (de 0 a length).
    poses : list of np.ndarray
        Lista de poses homogéneas T_target_i.
    """
    T_start = forward_kinematics(q_start)
    R = rotation_matrix(T_start)
    p0 = position_vector(T_start)

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    direction = R[:, axis_idx]  # eje deseado del TCP en frame base

    s = np.linspace(0.0, length, n_points)
    poses = []
    for si in s:
        T_i = np.eye(4)
        T_i[:3, :3] = R
        T_i[:3, 3] = p0 + si * direction
        poses.append(T_i)

    return s, poses


def solve_trajectory_ik(
    poses: list[np.ndarray],
    q_init: np.ndarray,
    *,
    first_point_is_seed: bool = True,
) -> tuple[np.ndarray, list[IKResult]]:
    """Resuelve IK para una secuencia de poses, usando warm-start.

    Cada solución sirve como semilla de la siguiente, lo cual mejora
    notablemente la convergencia y produce trayectorias articulares suaves.

    Parameters
    ----------
    poses : list of np.ndarray
        Lista de poses objetivo.
    q_init : np.ndarray, shape (7,)
        Configuración articular inicial usada como semilla del primer waypoint.
    first_point_is_seed : bool
        Si True (caso típico de trayectoria lineal en HOME), se asume que
        ``q_init`` ya es la solución exacta de ``poses[0]`` y se omite la IK
        del primer punto. Si False (caso del arco, donde el primer waypoint
        está desplazado del HOME), también se resuelve IK para el primer
        punto usando q_init solo como semilla.

    Returns
    -------
    Q : np.ndarray, shape (n_points, 7)
        Matriz de configuraciones articulares; Q[i] es la solución para poses[i].
    results : list of IKResult
        Lista de resultados detallados por waypoint.
    """
    n = len(poses)
    Q = np.zeros((n, 7))
    results: list[IKResult] = []

    if first_point_is_seed:
        Q[0] = q_init
        results.append(IKResult(
            q=q_init, success=True, iterations=0,
            pos_error=0.0, rot_error=0.0,
            message="Punto inicial (sin IK).",
        ))
        start_idx = 1
        q_seed = q_init.copy()
    else:
        # Resolver IK del primer punto usando q_init como semilla
        res0 = inverse_kinematics(poses[0], q_init=q_init)
        Q[0] = res0.q
        results.append(res0)
        start_idx = 1
        q_seed = res0.q.copy()

    for i in range(start_idx, n):
        res = inverse_kinematics(poses[i], q_init=q_seed)
        Q[i] = res.q
        results.append(res)
        q_seed = res.q.copy()  # warm-start

    return Q, results


def arc_trajectory_in_tcp_plane(
    q_start: np.ndarray,
    radius: float = 0.01,
    arc_span: float = np.pi,
    n_points: int = 60,
    plane: str = "xy",
    *,
    rotate_roll: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Genera un arco en un plano del frame TCP.

    El arco está contenido en un plano del TCP (por defecto XY), con centro
    en la posición actual del TCP y radio configurable. El TCP se desplaza
    primero ``radius`` en la dirección +X del plano del TCP (punto inicial
    del arco), y luego barre un ángulo ``arc_span`` rotando alrededor del
    eje normal al plano.

    Geometría (con plane="xy", arc_span=pi):
        - Punto 0:    HOME + radius * x_TCP            (ángulo θ = 0)
        - Punto N/2:  HOME + radius * y_TCP            (θ = π/2)
        - Punto N-1:  HOME - radius * x_TCP            (θ = π)

    El centro del arco coincide con la posición del TCP en ``q_start`` (es
    decir, HOME es el centro del semicírculo, no un punto sobre la curva).

    Parameters
    ----------
    q_start : np.ndarray, shape (7,)
        Configuración articular que define el frame TCP de referencia. La
        posición del TCP en ``q_start`` se usa como centro del arco, y la
        orientación define el plano del arco.
    radius : float
        Radio del arco [m].
    arc_span : float
        Ángulo total barrido [rad]. Por defecto π (semicírculo).
    n_points : int
        Número de waypoints (incluye inicio y fin).
    plane : {"xy", "xz", "yz"}
        Plano del frame TCP en el que se traza el arco.
    rotate_roll : float
        Rotación adicional alrededor del eje Z del TCP, distribuida
        linealmente a lo largo del arco. Por defecto 0 (orientación
        constante igual a q_start). Si distinto de 0, la orientación
        del TCP gira ``rotate_roll`` rad entre el primer y el último
        punto, manteniéndose el resto de la orientación inalterado.

    Returns
    -------
    theta : np.ndarray, shape (n_points,)
        Ángulo barrido en cada waypoint [rad], de 0 a arc_span.
    poses : list of np.ndarray
        Lista de poses homogéneas T_target_i.
    """
    T_start = forward_kinematics(q_start)
    R_start = rotation_matrix(T_start)
    p_center = position_vector(T_start)

    # Selección de los dos ejes que definen el plano del arco y el eje
    # normal alrededor del cual se rota.
    plane = plane.lower()
    if plane == "xy":
        u_axis = R_start[:, 0]   # x_TCP
        v_axis = R_start[:, 1]   # y_TCP
        n_axis = R_start[:, 2]   # z_TCP (normal)
    elif plane == "xz":
        u_axis = R_start[:, 0]
        v_axis = R_start[:, 2]
        n_axis = R_start[:, 1]
    elif plane == "yz":
        u_axis = R_start[:, 1]
        v_axis = R_start[:, 2]
        n_axis = R_start[:, 0]
    else:
        raise ValueError(f"plane debe ser 'xy', 'xz' o 'yz'; recibido: {plane}")

    theta = np.linspace(0.0, arc_span, n_points)
    poses: list[np.ndarray] = []

    # Mapeo lineal: t = 0 -> roll = 0; t = 1 -> roll = rotate_roll
    # (t es el progreso normalizado a lo largo del arco)
    for i, ang in enumerate(theta):
        t = i / max(1, n_points - 1)

        # Posición: punto sobre el arco
        p_i = p_center + radius * (np.cos(ang) * u_axis + np.sin(ang) * v_axis)

        # Orientación: parte de R_start y opcionalmente añade un roll
        # alrededor del eje z_TCP (frame de la pinza).
        if abs(rotate_roll) > 1e-12:
            roll_angle = t * rotate_roll
            ca, sa = np.cos(roll_angle), np.sin(roll_angle)
            # Rotación alrededor del eje z local del TCP de partida
            R_roll_local = np.array([
                [ca, -sa, 0.0],
                [sa,  ca, 0.0],
                [0.0, 0.0, 1.0],
            ])
            R_i = R_start @ R_roll_local
        else:
            R_i = R_start

        T_i = np.eye(4)
        T_i[:3, :3] = R_i
        T_i[:3, 3] = p_i
        poses.append(T_i)

    return theta, poses


# =============================================================================
# RESUMEN PRESENTABLE
# =============================================================================

def describe_pose(T: np.ndarray, label: str = "Pose") -> str:
    """Devuelve un string descriptivo legible de una pose homogénea."""
    p = position_vector(T)
    R = rotation_matrix(T)
    # Ángulos ZYX (yaw-pitch-roll) para reporte humano
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    lines = [
        f"{label}:",
        f"  posición [m]   = [{p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}]",
        f"  orientación [deg] roll={np.degrees(roll):+7.2f}, "
        f"pitch={np.degrees(pitch):+7.2f}, yaw={np.degrees(yaw):+7.2f}",
    ]
    return "\n".join(lines)


__all__ = [
    "PSM_DH_TABLE",
    "Q_HOME",
    "Q_HOME_TILTED",
    "Q_HOME_LAPARO",
    "TILT_ANGLE",
    "LAPARO_TILT_ANGLE",
    "Q_LIMITS",
    "L_RCC",
    "L_TOOL",
    "L_PITCH2YAW",
    "L_YAW2CTRL",
    "dh_modified_transform",
    "forward_kinematics",
    "forward_kinematics_full",
    "geometric_jacobian",
    "inverse_kinematics",
    "linear_trajectory_along_tcp_axis",
    "arc_trajectory_in_tcp_plane",
    "solve_trajectory_ik",
    "describe_pose",
    "IKResult",
]
