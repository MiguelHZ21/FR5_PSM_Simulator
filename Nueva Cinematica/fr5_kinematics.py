#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ***************************************************************
# Autores:    Miguel Hernandez (miguelhernandez@unicauca.edu.co)
#             Cristian Gonzalez (cgonzalezg@unicauca.edu.co)
# ***************************************************************
"""
fr5_kinematics.py — Cinematica directa e INVERSA CERRADA del FAIRINO FR5
========================================================================

Reemplazo completo de `fr5_ik` (MATLAB) para el robot descrito en `fr5v6.urdf`.

QUE HACE
--------
Resuelve la cinematica inversa de forma ANALITICA (no iterativa, sin semilla
obligatoria, sin convergencia que falle): devuelve **las 8 soluciones** del
robot para cualquier pose alcanzable, mas todas sus variantes equivalentes por
+/-360 deg que caen dentro de los limites articulares, ordenadas por cercania a
la posicion actual del robot.

POR QUE FUNCIONA (el hallazgo importante)
-----------------------------------------
El FR5 NO es un UR5 — las medidas son distintas — pero su ESTRUCTURA cinematica
es identica a la de la serie UR: hombro + 2 eslabones planos + muneca de 3 ejes
con OFFSET (no esferica, los 3 ultimos ejes no se cortan en un punto).

Desarrollando el URDF joint por joint:

    T = Rz(q1)·Tz(d1)·Rx(pi/2)·Rz(q2)·Tx(a2)·Rz(q3)·Tx(a3)
          ·Rz(q4)·Tz(d4)·Rx(pi/2)·Rz(q5)·Tz(d5)·Rx(-pi/2)·Rz(q6)·Tz(d6)

que es EXACTAMENTE el producto de Denavit-Hartenberg clasico
T_i = Rz(theta_i)·Tz(d_i)·Tx(a_i)·Rx(alpha_i) con la tabla:

    i | theta | d       | a        | alpha
    --+-------+---------+----------+--------
    1 | q1    | 0.152   |  0       |  pi/2
    2 | q2    | 0       | -0.425   |  0
    3 | q3    | 0       | -0.39501 |  0
    4 | q4    | 0.1021  |  0       |  pi/2
    5 | q5    | 0.102   |  0       | -pi/2
    6 | q6    | 0.457   |  0       |  0

=> El cero del URDF coincide con el cero de DH: NO hay offsets de junta.
   (verificado numericamente en selftest(): fk() vs fk_urdf())

Por lo tanto aplica la solucion cerrada tipo UR (muneca con offset), que da
2 (hombro izq/der) x 2 (muneca arriba/abajo) x 2 (codo arriba/abajo) = 8 ramas.

UNIDADES
--------
El nucleo trabaja en METROS y RADIANES (sin ambiguedad).
Los wrappers `*_fairino` trabajan en MILIMETROS y GRADOS, que es lo que usa el
controlador FAIRINO y lo que esperaba el script de MATLAB.

CONVENCION DE ORIENTACION — CONFIRMADA contra fr5_ik.m
-------------------------------------------------------
rx, ry, rz = RPY de ejes FIJOS X-Y-Z, o sea  R = Rz(rz)·Ry(ry)·Rx(rx).

No es una suposicion: `fr5_ik.m` linea 30 hace
    tform = trvec2tform([x y z]) * eul2tform([oriz, oriy, orix]);
y `eul2tform` con su secuencia por defecto "ZYX" da R = Rz(a1)·Ry(a2)·Rx(a3).
Pasando [oriz, oriy, orix] queda R = Rz(rz)·Ry(ry)·Rx(rx) — identico a esto.
(La variante comentada en la linea 31, eul2tform([orix,oriy,oriz]), SI seria
incorrecta: daria Rz(rx)·Ry(ry)·Rx(rz). Bien descartada.)

QUE HACE fr5_ik.m Y EN QUE SE DIFERENCIA ESTO
----------------------------------------------
fr5_ik.m usa `inverseKinematics` de MATLAB = solver NUMERICO iterativo, con
`AllowRandomRestart = false` y semilla FIJA [0,-90,90,-90,-90,90] deg en CADA
llamada (lineas 35-41; la version que arrastraba la solucion previa quedo
comentada al final del archivo). Consecuencias:
  · Es determinista (misma entrada -> misma salida). Eso esta bien.
  · Devuelve UNA sola de las hasta 8 soluciones: la del pozo de atraccion de esa
    semilla. Toda la mitad "espejo" del espacio de trabajo queda inaccesible.
  · Si no converge devuelve NaN(1,6) — que tiene largo 6 y por lo tanto PASA el
    chequeo `length(hi)~=6` del script principal.
  · No garantiza continuidad entre puntos vecinos de una trayectoria.
  · `isPoseReachable` usa link_lengths=[0.425,0.395,0.109,0.100] (d4 y d6 NO
    coinciden con este URDF: son 0.1021 y 0.267) y mide la distancia desde el
    ORIGEN en vez de desde el hombro (z=0.152). Descarta poses alcanzables y
    acepta inalcanzables.
Este modulo es cerrado (no itera, no falla por convergencia), devuelve TODAS las
soluciones y elige la mas cercana a donde el robot esta.

VERIFICAR CONTRA EL ROBOT REAL (hacelo una vez, antes de mover nada)
--------------------------------------------------------------------
    python fr5_kinematics.py --verificar  
        --q   0,-90,90,-90,-90,0          
        --tcp -500,0,400,180,0,0

Toma una lectura simultanea de juntas y de TCP publicada por el robot real y te
dice si el modelo (d6 y convencion RPY) coincide.

El TCP por defecto es la BRIDA (d6 = 0.100 m): el `tool_Link` del URDF es una
herramienta adicional y NO entra en la cinematica. Si el error de posicion sale
grande pero el de orientacion casi cero, el que no cuadra es d6: probá
`--d6 0.267` (con herramienta) o el valor que tenga configurado tu controlador.

Autor: generado para Angel Garzon. Sin dependencias fuera de numpy.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# =============================================================================
# 1. PARAMETROS DEL ROBOT
# =============================================================================

# --- Longitudes (metros), leidas del URDF fr5v6.urdf ---
D1 = 0.152      # base -> eje j2      (URDF: joint j2, xyz z)
A2 = -0.425     # eje j2 -> eje j3    (URDF: joint j3, xyz x)
A3 = -0.39501   # eje j3 -> eje j4    (URDF: joint j4, xyz x)
D4 = 0.1021     # offset lateral      (URDF: joint j5, xyz z)
D5 = 0.102      # offset de muneca    (URDF: joint j6, xyz z)

# d6 = distancia desde el origen del j6_Link hasta el punto que estas comandando,
# medida sobre el eje z6.
#
# DECISION (Angel, 2026-08-11): el `tool_Link` del URDF es una herramienta
# ADICIONAL y NO entra en la cinematica. El TCP es la BRIDA.
#
#   D6_BRIDA = 0.100  <- DEFECTO. Brida desnuda del FR5.
#   D6_TOOL_URDF = 0.457  <- lo que dice el URDF (brida 0.100 + herramienta 0.167)
#
# CUIDADO: los 0.100 salen de la ficha tecnica del FR5, NO del URDF (este URDF no
# tiene link de brida; fusiona brida + herramienta en `tool_Link` a 0.267). Es el
# unico numero de todo el modelo que no pude leer del URDF. Confirmalo con
# `--verificar` contra una lectura real ANTES de mover el robot. Si tu controlador
# tiene un TCP configurado, d6 tiene que coincidir con ESE punto.
D6_BRIDA = 0.100
D6_TOOL_URDF = 0.457


@dataclass(frozen=True)
class ParamsFR5:
    """Tabla DH del FR5. Cambiar d6 si tu TCP es otro."""
    d1: float = D1
    a2: float = A2
    a3: float = A3
    d4: float = D4
    d5: float = D5
    d6: float = D6_TOOL_URDF

    def tabla_dh(self) -> list[tuple[float, float, float]]:
        """[(d, a, alpha)] por junta, en orden 1..6."""
        return [
            (self.d1, 0.0, math.pi / 2),
            (0.0, self.a2, 0.0),
            (0.0, self.a3, 0.0),
            (self.d4, 0.0, math.pi / 2),
            (self.d5, 0.0, -math.pi / 2),
            (self.d6, 0.0, 0.0),
        ]


FR5 = ParamsFR5()                                    # TCP = BRIDA (0.100 m por defecto)
FR5_CON_HERRAMIENTA = ParamsFR5(d6=D6_TOOL_URDF)     # TCP = tool_Link del URDF (0.457 m)


# --- Limites articulares (radianes), leidos literal del URDF ---
LIMITES = np.array([
    [-3.0543,  3.0543],   # j1  +/-175.0 deg
    [-4.6251,  1.4835],   # j2  -265.0 .. +85.0 deg
    [-2.8274,  2.8274],   # j3  +/-162.0 deg
    [-4.6251,  1.4835],   # j4  -265.0 .. +85.0 deg
    [-3.0543,  3.0543],   # j5  +/-175.0 deg
    [-3.0543,  3.0543],   # j6  +/-175.0 deg
])

# Peso de cada junta al elegir "la solucion mas parecida a donde estoy".
# Mover j1 (que arrastra todo el brazo) cuesta mas que mover j6 (la muneca).
PESOS = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

RPY_CONVENCION = "xyz_fijo"   # R = Rz(rz)·Ry(ry)·Rx(rx)  (URDF / ROS / FAIRINO)

_EPS_UNIDAD = 1e-9      # margen al saturar acos/asin
_EPS_SINGULAR = 1e-7    # |sin(q5)| por debajo de esto => muneca singular
_AVISO_SINGULAR = 1e-3  # |sin(q5)| por debajo de esto => cerca de singular


# =============================================================================
# 2. UTILIDADES DE TRANSFORMACIONES
# =============================================================================

def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matriz DH clasica:  Rz(theta)·Tz(d)·Tx(a)·Rx(alpha)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inversa exacta de una transformacion homogenea (sin resolver sistemas)."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def rpy_a_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """RPY de ejes fijos XYZ (rad) -> matriz de rotacion.  R = Rz·Ry·Rx."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,                    cy * sx,                cy * cx],
    ])


def R_a_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Matriz de rotacion -> RPY de ejes fijos XYZ (rad). Maneja gimbal lock."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(math.cos(ry)) < 1e-9:            # gimbal lock: ry = +/-90 deg
        rz = 0.0
        rx = math.atan2(-R[1, 2], R[1, 1])
    else:
        rz = math.atan2(R[1, 0], R[0, 0])
        rx = math.atan2(R[2, 1], R[2, 2])
    return rx, ry, rz


def pose_a_T(x, y, z, rx, ry, rz) -> np.ndarray:
    """(m, rad) -> matriz homogenea 4x4."""
    T = np.eye(4)
    T[:3, :3] = rpy_a_R(rx, ry, rz)
    T[:3, 3] = (x, y, z)
    return T


def T_a_pose(T: np.ndarray) -> tuple[float, ...]:
    """Matriz homogenea 4x4 -> (x, y, z, rx, ry, rz) en (m, rad)."""
    return (*T[:3, 3], *R_a_rpy(T[:3, :3]))


def error_pose(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """Error entre dos poses: (metros de posicion, radianes de orientacion)."""
    e_pos = float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))
    Rrel = T1[:3, :3].T @ T2[:3, :3]
    coseno = (np.trace(Rrel) - 1.0) / 2.0
    e_rot = float(math.acos(max(-1.0, min(1.0, coseno))))
    return e_pos, e_rot


# =============================================================================
# 3. CINEMATICA DIRECTA
# =============================================================================

def fk(q: Sequence[float], params: ParamsFR5 = FR5,
       hasta: int = 6) -> np.ndarray:
    """
    Cinematica directa por DH. q en radianes.
    `hasta`=k devuelve la transformacion base -> frame k (util para el jacobiano).
    """
    T = np.eye(4)
    for i, (d, a, alpha) in enumerate(params.tabla_dh()[:hasta]):
        T = T @ _dh(q[i], d, a, alpha)
    return T


def fk_urdf(q: Sequence[float]) -> np.ndarray:
    """
    Cinematica directa construida joint-por-joint DIRECTO del XML del URDF,
    sin pasar por DH. Existe solo para VERIFICAR que fk() es correcta.
    Ojo: el URDF escribe 1.5708 en vez de pi/2 -> hay ~3.7e-6 rad de diferencia
    de redondeo del exportador de SolidWorks (unos 4 um a 1 m). Es del URDF,
    no del modelo.
    """
    def Rz(t):
        T = np.eye(4); c, s = math.cos(t), math.sin(t)
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; return T

    def Rx(t):
        T = np.eye(4); c, s = math.cos(t), math.sin(t)
        T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]; return T

    def Tr(x, y, z):
        T = np.eye(4); T[:3, 3] = (x, y, z); return T

    T = np.eye(4)
    T = T @ Rz(q[0])                                          # base_link -> j1_Link
    T = T @ Tr(0, 0, 0.152) @ Rx(1.5708) @ Rz(q[1])           # j1 -> j2
    T = T @ Tr(-0.425, 0, 0) @ Rz(q[2])                       # j2 -> j3
    T = T @ Tr(-0.39501, 0, 0) @ Rz(q[3])                     # j3 -> j4
    T = T @ Tr(0, 0, 0.1021) @ Rx(1.5708) @ Rz(q[4])          # j4 -> j5
    T = T @ Tr(0, 0, 0.102) @ Rx(-1.5708) @ Rz(q[5])          # j5 -> j6
    T = T @ Tr(0, 0, 0.267)                                   # j6 -> tool_Link
    return T


def jacobiano(q: Sequence[float], params: ParamsFR5 = FR5) -> np.ndarray:
    """Jacobiano geometrico 6x6 en la base. Filas: [vx vy vz wx wy wz]."""
    Ts = [np.eye(4)]
    T = np.eye(4)
    for i, (d, a, alpha) in enumerate(params.tabla_dh()):
        T = T @ _dh(q[i], d, a, alpha)
        Ts.append(T.copy())
    p_e = Ts[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = Ts[i][:3, 2]
        p = Ts[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def distancia_a_singularidad(q: Sequence[float],
                             params: ParamsFR5 = FR5) -> float:
    """
    Menor valor singular del jacobiano. Cerca de 0 => configuracion singular
    (el robot pierde un grado de libertad y las velocidades se disparan).
    Regla practica: por debajo de ~0.01 no pases por ahi a velocidad.
    """
    return float(np.linalg.svd(jacobiano(q, params), compute_uv=False)[-1])


# =============================================================================
# 4. CINEMATICA INVERSA CERRADA
# =============================================================================

@dataclass
class SolucionIK:
    q: np.ndarray                 # 6 angulos en radianes
    rama: str                     # p.ej. "hombro=izq codo=arriba muneca=abajo"
    vueltas: tuple = (0,) * 6     # multiplos de 2*pi aplicados a cada junta
    costo: float = float("nan")   # distancia ponderada a la semilla
    singular: bool = False        # muneca en/cerca de singularidad
    params: ParamsFR5 = FR5
    _sigma: float | None = field(default=None, repr=False)

    @property
    def manipulabilidad(self) -> float:
        """
        Menor valor singular del jacobiano (distancia a la singularidad).
        PEREZOSA: el SVD solo se calcula la primera vez que la pedis. Calcularla
        para las hasta 32 candidatas de cada pose costaba ~20 veces mas que
        resolver la cinematica inversa entera.
        """
        if self._sigma is None:
            self._sigma = distancia_a_singularidad(self.q, self.params)
        return self._sigma

    def grados(self) -> np.ndarray:
        return np.degrees(self.q)


def _saturar(v: float) -> float:
    return max(-1.0, min(1.0, v))


def ik_ramas(T_objetivo: np.ndarray,
             params: ParamsFR5 = FR5,
             tool: np.ndarray | None = None,
             q_semilla: Sequence[float] | None = None) -> list[SolucionIK]:
    """
    Las hasta 8 ramas analiticas, SIN aplicar limites ni vueltas de +/-360.

    T_objetivo: pose deseada del TCP en la base, 4x4, metros/radianes.
    tool      : transformacion opcional frame6 -> TCP (si tu herramienta no es
                un simple offset sobre z6). Si la pasas, poné d6 = la parte
                sobre z6 que YA esta en la tabla DH y el resto en `tool`.
    q_semilla : solo se usa para desempatar q6 si la muneca cae singular.
    """
    d1, a2, a3 = params.d1, params.a2, params.a3
    d4, d5, d6 = params.d4, params.d5, params.d6

    if abs(d6) < 1e-9:
        raise ValueError(
            "d6 = 0: el TCP esta sobre el centro de muneca y q5 queda "
            "indeterminado. Usa un TCP con offset (p.ej. d6=0.267)."
        )

    T06 = T_objetivo if tool is None else T_objetivo @ inv_T(tool)
    p = T06[:3, 3]
    n, o, a = T06[:3, 0], T06[:3, 1], T06[:3, 2]
    q6_semilla = float(q_semilla[5]) if q_semilla is not None else 0.0

    soluciones: list[SolucionIK] = []

    # ---- q1: el centro del frame 5 debe quedar a distancia d4 del eje 1 -----
    p05 = p - d6 * a
    radio = math.hypot(p05[0], p05[1])
    if radio < abs(d4) - 1e-12:
        return []                      # singularidad de hombro / inalcanzable
    phi = math.atan2(p05[1], p05[0])
    delta = math.asin(_saturar(d4 / radio))

    for i_hombro, q1 in enumerate((phi + delta, phi + math.pi - delta)):
        s1, c1 = math.sin(q1), math.cos(q1)

        # ---- q5: proyeccion del TCP sobre el eje y del frame 1 -------------
        arg5 = (p[0] * s1 - p[1] * c1 - d4) / d6
        if abs(arg5) > 1.0 + 1e-6:
            continue
        arg5 = _saturar(arg5)

        for i_muneca, q5 in enumerate((math.acos(arg5), -math.acos(arg5))):
            s5 = math.sin(q5)

            # ---- q6: de la orientacion ------------------------------------
            num = -o[0] * s1 + o[1] * c1
            den = n[0] * s1 - n[1] * c1
            singular = abs(s5) < _EPS_SINGULAR or math.hypot(num, den) < 1e-10
            if singular:
                # Muneca singular (q5 ~ 0 o +/-180): j4 y j6 son coaxiales, q6
                # es libre. Fijamos q6 = el actual del robot y el resto lo
                # absorbe q4. Por eso conviene pasar q_semilla.
                q6 = q6_semilla
            else:
                sgn = 1.0 if s5 > 0.0 else -1.0
                q6 = math.atan2(sgn * num, sgn * den)

            # ---- q2, q3, q4: problema plano de 2 eslabones -----------------
            T01 = _dh(q1, d1, 0.0, math.pi / 2)
            T45 = _dh(q5, d5, 0.0, -math.pi / 2)
            T56 = _dh(q6, d6, 0.0, 0.0)
            T14 = inv_T(T01) @ T06 @ inv_T(T56) @ inv_T(T45)

            px, py = T14[0, 3], T14[1, 3]
            arg3 = (px * px + py * py - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if abs(arg3) > 1.0 + 1e-6:
                continue               # fuera de alcance para esta rama
            arg3 = _saturar(arg3)

            q234 = math.atan2(T14[1, 0], T14[0, 0])

            for i_codo, q3 in enumerate((math.acos(arg3), -math.acos(arg3))):
                q2 = math.atan2(py, px) - math.atan2(
                    a3 * math.sin(q3), a2 + a3 * math.cos(q3))
                q4 = q234 - q2 - q3
                q = _envolver(np.array([q1, q2, q3, q4, q5, q6]))
                rama = (f"hombro={'izq' if i_hombro == 0 else 'der'} "
                        f"muneca={'arriba' if i_muneca == 0 else 'abajo'} "
                        f"codo={'arriba' if i_codo == 0 else 'abajo'}")
                soluciones.append(SolucionIK(q=q, rama=rama, singular=singular))

    return soluciones


def explicar(T_objetivo: np.ndarray,
             params: ParamsFR5 = FR5,
             tool: np.ndarray | None = None) -> str:
    """
    Explica EN CASTELLANO por que una pose sale (o no) y cuantas ramas pierde.

    Existe porque el mensaje del MATLAB ("Valores de J4 o J5 fuera de limites")
    aparecia tambien cuando la pose era simplemente inalcanzable, y mandaba a
    buscar el problema al lado equivocado.

    Los tres motivos geometricos posibles son:
      · cilindro muerto : el centro de muneca cae a menos de d4 del eje de j1
      · demasiado lejos : el subconjunto plano 2R necesita mas de |a2|+|a3|
      · demasiado cerca : necesita menos de ||a2|-|a3||, el codo no pliega tanto
    """
    d1, a2, a3, d4, d6 = (params.d1, params.a2, params.a3, params.d4, params.d6)
    T06 = T_objetivo if tool is None else T_objetivo @ inv_T(tool)
    p, a = T06[:3, 3], T06[:3, 2]
    p05 = p - d6 * a
    radio = math.hypot(p05[0], p05[1])
    r_max, r_min = abs(a2) + abs(a3), abs(abs(a2) - abs(a3))

    lineas = [
        f"TCP        : x={p[0]*1000:.1f} y={p[1]*1000:.1f} z={p[2]*1000:.1f} mm",
        f"centro de muneca a {radio*1000:.1f} mm del eje de j1 "
        f"(minimo d4 = {d4*1000:.1f} mm)",
        f"alcance del subconjunto 2R: {r_min*1000:.1f} .. {r_max*1000:.1f} mm",
    ]
    if radio < abs(d4):
        lineas.append("=> INALCANZABLE: el TCP cae dentro del cilindro muerto "
                      "alrededor del eje de j1.")
        return "\n".join(lineas)

    ramas = ik_ramas(T06, params=params)
    validas = ik(T_objetivo, params=params, tool=tool)
    lineas.append(f"ramas geometricas: {len(ramas)} de 8 "
                  f"(las que faltan no existen: piden un alcance 2R fuera de "
                  f"[{r_min*1000:.1f}, {r_max*1000:.1f}] mm)")
    lineas.append(f"soluciones dentro de limites articulares: {len(validas)}")
    if not ramas:
        d_hombro = math.hypot(radio, p05[2] - d1)
        lineas.append(
            "=> INALCANZABLE: " +
            ("demasiado LEJOS" if d_hombro > r_max else "demasiado CERCA") +
            f" (centro de muneca a {d_hombro*1000:.1f} mm del hombro).")
    elif not validas:
        lineas.append("=> La pose es geometricamente alcanzable pero NINGUNA "
                      "rama entra en los limites articulares del URDF.")
    else:
        lineas.append(f"=> OK. Mejor rama: {validas[0].rama}, "
                      f"sigma_min={validas[0].manipulabilidad:.4f}")
    return "\n".join(lineas)


def _envolver(q: np.ndarray) -> np.ndarray:
    """Lleva cada angulo al intervalo (-pi, pi]."""
    return np.array([(-((-x + math.pi) % (2 * math.pi)) + math.pi) for x in q])


def _variantes_por_vueltas(q: np.ndarray,
                           limites: np.ndarray) -> list[tuple[np.ndarray, tuple]]:
    """
    j2 y j4 del FR5 recorren 350 deg (-265..+85), asi que MUCHOS angulos tienen
    DOS representaciones validas: q y q-360. El script de MATLAB no lo
    contemplaba y rechazaba puntos perfectamente alcanzables.
    Devuelve todas las combinaciones que caen dentro de limites.
    """
    por_junta = []
    for i in range(6):
        cand = []
        for k in (-1, 0, 1):
            v = q[i] + 2 * math.pi * k
            if limites[i, 0] - 1e-9 <= v <= limites[i, 1] + 1e-9:
                cand.append((v, k))
        if not cand:
            return []                  # esta junta no entra de ninguna forma
        por_junta.append(cand)

    salida = []
    for combo in itertools.product(*por_junta):
        salida.append((np.array([c[0] for c in combo]),
                       tuple(c[1] for c in combo)))
    return salida


def ik(T_objetivo: np.ndarray,
       q_semilla: Sequence[float] | None = None,
       params: ParamsFR5 = FR5,
       tool: np.ndarray | None = None,
       limites: np.ndarray | None = LIMITES,
       pesos: np.ndarray = PESOS,
       verificar: bool = True,
       tol_pos: float = 1e-6,
       tol_rot: float = 1e-6) -> list[SolucionIK]:
    """
    CINEMATICA INVERSA COMPLETA.

    Devuelve TODAS las soluciones validas (8 ramas x variantes de +/-360 deg),
    ya filtradas por limites articulares, deduplicadas, VERIFICADAS con
    cinematica directa, y ordenadas de la mas cercana a `q_semilla` a la mas
    lejana. Si no pasas semilla, ordena por cercania a cero.

    Lista vacia = pose inalcanzable (o fuera de limites). Nunca devuelve
    NaN, complejos, ni una solucion que no reproduzca la pose pedida.
    """
    crudas = ik_ramas(T_objetivo, params=params, tool=tool, q_semilla=q_semilla)

    semilla = (np.asarray(q_semilla, dtype=float)
               if q_semilla is not None else np.zeros(6))

    vistas: set[tuple] = set()
    salida: list[SolucionIK] = []

    for sol in crudas:
        if not np.all(np.isfinite(sol.q)):
            continue

        # Verificacion por RAMA, no por variante: sumar +/-2*pi a una junta no
        # cambia sin/cos, asi que todas las variantes de una rama dan
        # EXACTAMENTE la misma cinematica directa. Verificar una alcanza.
        if verificar:
            T_obt = fk(sol.q, params)
            if tool is not None:
                T_obt = T_obt @ tool
            e_pos, e_rot = error_pose(T_objetivo, T_obt)
            if e_pos > tol_pos or e_rot > tol_rot:
                continue               # rama espuria: se descarta, no se manda

        variantes = ([(sol.q, (0,) * 6)] if limites is None
                     else _variantes_por_vueltas(sol.q, limites))
        for q_var, vueltas in variantes:
            clave = tuple(np.round(q_var, 9))
            if clave in vistas:
                continue
            vistas.add(clave)

            d = q_var - semilla
            salida.append(SolucionIK(
                q=q_var,
                rama=sol.rama,
                vueltas=vueltas,
                costo=float(np.sqrt(np.sum(pesos * d * d))),
                singular=sol.singular,
                params=params,
            ))

    salida.sort(key=lambda s: s.costo)
    return salida


def ik_mejor(T_objetivo: np.ndarray,
             q_semilla: Sequence[float] | None = None,
             salto_max: float | None = math.radians(90.0),
             **kw) -> SolucionIK | None:
    """
    La mejor solucion: valida, dentro de limites y lo mas parecida posible a
    donde el robot ESTA AHORA.

    `salto_max`: si la solucion elegida obliga a mover alguna junta mas que
    esto respecto de la semilla, se devuelve None. Es la red de seguridad
    contra el salto de rama (el bug que te movia el robot 180 deg entre dos
    puntos cartesianos "vecinos"). Poné None para desactivarla.
    """
    sols = ik(T_objetivo, q_semilla=q_semilla, **kw)
    if not sols:
        return None
    mejor = sols[0]
    if salto_max is not None and q_semilla is not None:
        if np.max(np.abs(mejor.q - np.asarray(q_semilla, float))) > salto_max:
            return None
    return mejor


def ik_trayectoria(poses: Iterable[np.ndarray],
                   q_inicial: Sequence[float],
                   salto_max: float | None = math.radians(45.0),
                   **kw) -> tuple[np.ndarray, list[str]]:
    """
    Resuelve una lista completa de poses ENCADENANDO la continuidad: cada punto
    usa como semilla la solucion del punto anterior. Es lo que le faltaba al
    script de MATLAB — sin esto, dos waypoints vecinos pueden caer en ramas
    distintas y el robot pega un latigazo.

    Devuelve (matriz Nx6 en radianes, lista de problemas). Si un punto falla,
    su fila queda en NaN y se anota el motivo; los siguientes siguen usando la
    ultima semilla buena.
    """
    q_prev = np.asarray(q_inicial, dtype=float).copy()
    filas, problemas = [], []
    for i, T in enumerate(poses):
        sol = ik_mejor(T, q_semilla=q_prev, salto_max=salto_max, **kw)
        if sol is None:
            filas.append(np.full(6, np.nan))
            problemas.append(
                f"punto {i}: sin solucion valida "
                f"(inalcanzable, fuera de limites, o salto > "
                f"{math.degrees(salto_max):.0f} deg)" if salto_max
                else f"punto {i}: sin solucion valida")
            continue
        if sol.manipulabilidad < 0.01:
            problemas.append(
                f"punto {i}: cerca de singularidad (sigma_min="
                f"{sol.manipulabilidad:.4f}) — bajá la velocidad")
        filas.append(sol.q)
        q_prev = sol.q
    return np.array(filas), problemas


# =============================================================================
# 5. INTERFAZ FAIRINO (milimetros y grados) — reemplazo directo de fr5_ik
# =============================================================================

def fk_fairino(q_grados: Sequence[float],
               params: ParamsFR5 = FR5) -> tuple[float, ...]:
    """q en grados -> (x, y, z, rx, ry, rz) en mm y grados."""
    T = fk(np.radians(q_grados), params)
    x, y, z, rx, ry, rz = T_a_pose(T)
    return (x * 1000.0, y * 1000.0, z * 1000.0,
            math.degrees(rx), math.degrees(ry), math.degrees(rz))


def ik_fairino(x, y, z, rx, ry, rz,
               q_actual_grados: Sequence[float] | None = None,
               params: ParamsFR5 = FR5,
               salto_max_grados: float | None = 90.0,
               todas: bool = False):
    """
    Reemplazo directo de `fr5_ik(x,y,z,rx,ry,rz)` del MATLAB.

    Entrada : mm y grados (lo que usa el controlador FAIRINO).
    Salida  : lista de 6 grados (la mejor solucion), o None si es inalcanzable.
              Con todas=True devuelve la lista completa de SolucionIK.

    A diferencia del original: nunca devuelve NaN ni complejos, respeta los 6
    limites del URDF (no solo j4/j5), prueba las representaciones +/-360, y si
    le pasas la posicion actual del robot elige la solucion mas cercana.
    """
    T = pose_a_T(x / 1000.0, y / 1000.0, z / 1000.0,
                 math.radians(rx), math.radians(ry), math.radians(rz))
    semilla = np.radians(q_actual_grados) if q_actual_grados is not None else None

    if todas:
        return ik(T, q_semilla=semilla, params=params)

    salto = (math.radians(salto_max_grados)
             if (salto_max_grados is not None and semilla is not None) else None)
    sol = ik_mejor(T, q_semilla=semilla, salto_max=salto, params=params)
    return None if sol is None else sol.grados().tolist()


def verificar_contra_robot(q_grados: Sequence[float],
                           tcp_mm_grados: Sequence[float],
                           params: ParamsFR5 = FR5) -> dict:
    """
    Comparalo con una lectura REAL simultanea del robot (juntas + TCP que
    publica el controlador). Si el error es grande, el modelo no coincide.

    Interpretacion rapida del resultado:
      - error de posicion < 1 mm y de orientacion < 0.1 deg -> el modelo es correcto.
      - error de posicion grande, orientacion casi 0  -> d6 mal (ajusta d6).
      - error de orientacion grande                    -> convencion RPY distinta.
    """
    T_modelo = fk(np.radians(q_grados), params)
    T_robot = pose_a_T(tcp_mm_grados[0] / 1000.0,
                       tcp_mm_grados[1] / 1000.0,
                       tcp_mm_grados[2] / 1000.0,
                       math.radians(tcp_mm_grados[3]),
                       math.radians(tcp_mm_grados[4]),
                       math.radians(tcp_mm_grados[5]))
    e_pos, e_rot = error_pose(T_robot, T_modelo)
    delta = T_modelo[:3, 3] - T_robot[:3, 3]
    return {
        "error_posicion_mm": e_pos * 1000.0,
        "error_orientacion_deg": math.degrees(e_rot),
        "delta_xyz_mm": (delta * 1000.0).round(3).tolist(),
        "d6_usado_m": params.d6,
        "modelo_ok": bool(e_pos < 1e-3 and e_rot < math.radians(0.1)),
    }


# =============================================================================
# 6. AUTOTEST — corré esto antes de confiar en nada
# =============================================================================

def selftest(n: int = 4000, semilla_rng: int = 20260811, verboso: bool = True) -> bool:
    rng = np.random.default_rng(semilla_rng)
    ok = True

    def linea(t):
        if verboso: print(t)

    linea("=" * 72)
    linea("AUTOTEST fr5_kinematics.py")
    linea("=" * 72)

    # --- 1. La tabla DH reproduce el URDF ---------------------------------
    # fk_urdf() llega hasta `tool_Link`, asi que hay que compararla contra el
    # modelo CON herramienta (d6=0.267), no contra el de brida (d6=0.100).
    # De paso, la diferencia entre ambos tiene que dar los 167 mm de la
    # herramienta que el URDF trae fusionada.
    peor_p = peor_r = 0.0
    for _ in range(500):
        q = rng.uniform(LIMITES[:, 0], LIMITES[:, 1])
        ep, er = error_pose(fk_urdf(q), fk(q, FR5_CON_HERRAMIENTA))
        peor_p, peor_r = max(peor_p, ep), max(peor_r, er)
    linea(f"\n[1] DH vs URDF crudo (500 poses aleatorias)")
    linea(f"    posicion max    : {peor_p*1e6:9.3f} um")
    linea(f"    orientacion max : {math.degrees(peor_r)*3600:9.3f} arcseg")
    linea( "    (residuo esperado: el URDF escribe 1.5708 en vez de pi/2)")
    if peor_p > 5e-5:
        ok = False; linea("    !! DIFERENCIA MAYOR A LA ESPERADA")

    q_ref = np.radians([15.0, -80.0, 70.0, -60.0, -85.0, 20.0])
    salto = float(np.linalg.norm(fk(q_ref, FR5_CON_HERRAMIENTA)[:3, 3]
                                 - fk(q_ref, FR5)[:3, 3]))
    linea(f"    brida -> tool_Link: {salto*1000:.3f} mm "
          f"(esperado {(D6_TOOL_URDF-D6_BRIDA)*1000:.3f} = la herramienta)")
    if abs(salto - (D6_TOOL_URDF - D6_BRIDA)) > 1e-9:
        ok = False; linea("    !! el offset de herramienta no cuadra")

    # --- 2. Ida y vuelta: q -> FK -> IK -> q ------------------------------
    linea(f"\n[2] Ida y vuelta sobre {n} configuraciones aleatorias")
    fallos_recup = fallos_pose = sin_sol = 0
    peor_pos = peor_rot = 0.0
    hist = {}
    for _ in range(n):
        q = rng.uniform(LIMITES[:, 0], LIMITES[:, 1])
        T = fk(q, FR5)
        sols = ik(T, q_semilla=q, params=FR5)
        if not sols:
            sin_sol += 1
            continue
        hist[len(sols)] = hist.get(len(sols), 0) + 1
        for s in sols:                                  # toda solucion debe valer
            ep, er = error_pose(T, fk(s.q, FR5))
            peor_pos, peor_rot = max(peor_pos, ep), max(peor_rot, er)
            if ep > 1e-6 or er > 1e-6:
                fallos_pose += 1
        if not any(np.allclose(s.q, q, atol=1e-6) for s in sols):
            fallos_recup += 1
    linea(f"    sin solucion            : {sin_sol}")
    linea(f"    soluciones que NO cierran: {fallos_pose}")
    linea(f"    q original no recuperada : {fallos_recup}")
    linea(f"    error max posicion       : {peor_pos*1e9:9.3f} nm")
    linea(f"    error max orientacion    : {math.degrees(peor_rot)*3.6e6:9.3f} udeg")
    linea(f"    soluciones por pose      : "
          f"{dict(sorted(hist.items()))}")
    if sin_sol or fallos_pose or fallos_recup:
        ok = False; linea("    !! FALLO")

    # --- 3. Las 8 ramas en una pose interior ------------------------------
    # OJO: 8 ramas es el maximo, no una garantia. Cerca de la extension total
    # del brazo, invertir la muneca (q5 -> -q5) desplaza el centro de muneca
    # mas alla de |a2|+|a3| = 0.820 m y esa rama deja de existir DE VERDAD.
    # Que salgan menos de 8 ahi es correcto, no un fallo.
    q_comodo = np.radians([20.0, -100.0, 80.0, -70.0, -90.0, 30.0])
    ramas = ik_ramas(fk(q_comodo, FR5), FR5)
    linea(f"\n[3] Ramas analiticas en una pose bien interior: {len(ramas)} "
          f"(esperado 8)")
    for r in ramas:
        linea(f"    {r.rama:42s} {np.round(np.degrees(r.q), 2)}")
    if len(ramas) != 8:
        ok = False; linea("    !! No salieron las 8 ramas")

    # 3b. Cuando salen menos de 8, el motivo tiene que ser geometrico REAL.
    #     El subconjunto plano 2R solo alcanza el anillo
    #         ||a2|-|a3|| = 30.0 mm  ..  |a2|+|a3| = 820.0 mm
    #     Hay que mirar los DOS bordes: descartar una rama por estar demasiado
    #     CERCA (el codo no pliega tanto) es tan legitimo como por estar lejos.
    linea("\n[3b] Distribucion de ramas y verificacion del motivo de las que faltan")
    r_max = abs(FR5.a2) + abs(FR5.a3)
    r_min = abs(abs(FR5.a2) - abs(FR5.a3))
    reparto, sin_justificar, invariante_max = {}, 0, 0.0
    for _ in range(600):
        q = rng.uniform(LIMITES[:, 0], LIMITES[:, 1])
        T06 = fk(q, FR5)
        p, n_, o_, a_ = T06[:3, 3], T06[:3, 0], T06[:3, 1], T06[:3, 2]
        ramas = ik_ramas(T06, FR5)
        reparto[len(ramas)] = reparto.get(len(ramas), 0) + 1

        # Recorrer las 8 combos y comprobar una por una
        p05 = p - FR5.d6 * a_
        radio = math.hypot(p05[0], p05[1])
        if radio < abs(FR5.d4):
            continue
        phi = math.atan2(p05[1], p05[0])
        delta = math.asin(_saturar(FR5.d4 / radio))
        for q1 in (phi + delta, phi + math.pi - delta):
            s1, c1 = math.sin(q1), math.cos(q1)
            arg5 = (p[0] * s1 - p[1] * c1 - FR5.d4) / FR5.d6
            if abs(arg5) > 1.0 + 1e-6:
                continue                       # motivo valido: geometria de muneca
            for q5 in (math.acos(_saturar(arg5)), -math.acos(_saturar(arg5))):
                s5 = math.sin(q5)
                num, den = -o_[0] * s1 + o_[1] * c1, n_[0] * s1 - n_[1] * c1
                sgn = 1.0 if s5 > 0 else -1.0
                q6 = math.atan2(sgn * num, sgn * den)
                T14 = (inv_T(_dh(q1, FR5.d1, 0.0, math.pi / 2)) @ T06
                       @ inv_T(_dh(q6, FR5.d6, 0.0, 0.0))
                       @ inv_T(_dh(q5, FR5.d5, 0.0, -math.pi / 2)))
                # invariante de la derivacion: la z de T14 SIEMPRE vale d4
                invariante_max = max(invariante_max, abs(T14[2, 3] - FR5.d4))
                alcance = math.hypot(T14[0, 3], T14[1, 3])
                fuera = alcance > r_max + 1e-9 or alcance < r_min - 1e-9
                arg3 = ((alcance ** 2 - FR5.a2 ** 2 - FR5.a3 ** 2)
                        / (2.0 * FR5.a2 * FR5.a3))
                if abs(arg3) > 1.0 + 1e-6 and not fuera:
                    sin_justificar += 1        # descartada sin razon geometrica
    linea(f"    ramas por pose : {dict(sorted(reparto.items()))}")
    linea(f"    anillo 2R      : {r_min*1000:.1f} .. {r_max*1000:.1f} mm")
    linea(f"    invariante |p14_z - d4| max: {invariante_max:.3e} m  (debe ser ~0)")
    linea(f"    ramas descartadas SIN motivo geometrico: {sin_justificar}")
    if sin_justificar or invariante_max > 1e-12:
        ok = False; linea("    !! FALLO")

    # --- 4. Muneca singular (q5 = 0 y q5 = 180) ---------------------------
    linea("\n[4] Muneca singular (q5 = 0 y q5 = 180 deg)")
    for q5 in (0.0, math.pi):
        q = np.array([0.3, -1.2, 1.0, -1.4, q5, 0.7])
        T = fk(q, FR5)
        sols = ik(T, q_semilla=q, params=FR5)
        if not sols:
            ok = False; linea(f"    q5={math.degrees(q5):6.1f} -> !! sin solucion")
            continue
        ep, er = error_pose(T, fk(sols[0].q, FR5))
        marca = "OK" if (ep < 1e-6 and er < 1e-6) else "!! FALLO"
        linea(f"    q5={math.degrees(q5):6.1f} -> {len(sols):2d} sol, "
              f"err {ep*1e9:6.2f} nm / {math.degrees(er)*3.6e6:6.2f} udeg, "
              f"detectada singular={sols[0].singular}  {marca}")
        if ep > 1e-6 or er > 1e-6:
            ok = False

    # --- 5. Pose inalcanzable -> lista vacia, no basura -------------------
    T_lejos = pose_a_T(2.5, 0.0, 0.5, math.pi, 0.0, 0.0)
    sols = ik(T_lejos, params=FR5)
    linea(f"\n[5] Pose a 2.5 m (fuera de alcance): {len(sols)} soluciones "
          f"{'OK' if not sols else '!! deberia ser 0'}")
    if sols:
        ok = False

    # --- 6. Continuidad de trayectoria ------------------------------------
    linea("\n[6] Trayectoria continua (arco de 40 puntos)")
    q0 = np.radians([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    T0 = fk(q0, FR5)
    poses = []
    for k in range(40):
        T = T0.copy()
        T[0, 3] += 0.10 * math.sin(k / 39.0 * math.pi)
        T[1, 3] += 0.15 * (k / 39.0 - 0.5)
        poses.append(T)
    Q, probs = ik_trayectoria(poses, q0, params=FR5)
    salto = float(np.nanmax(np.abs(np.diff(Q, axis=0)))) if len(Q) > 1 else 0.0
    nan = int(np.isnan(Q).any(axis=1).sum())
    linea(f"    puntos sin resolver : {nan}")
    linea(f"    salto maximo entre puntos consecutivos: "
          f"{math.degrees(salto):.3f} deg")
    for p in probs:
        linea(f"    aviso: {p}")
    if nan or math.degrees(salto) > 10.0:
        ok = False; linea("    !! discontinuidad")

    # --- 7. Wrapper FAIRINO en mm/grados ----------------------------------
    linea("\n[7] Wrapper FAIRINO (mm/grados)")
    q_ref = [10.0, -70.0, 60.0, -80.0, -95.0, 25.0]
    pose = fk_fairino(q_ref, FR5)
    q_sol = ik_fairino(*pose, q_actual_grados=q_ref)
    linea(f"    q referencia : {np.round(q_ref, 3)}")
    linea(f"    pose (mm/deg): {np.round(pose, 3)}")
    linea(f"    q recuperada : {np.round(q_sol, 3) if q_sol else None}")
    if q_sol is None or np.max(np.abs(np.array(q_sol) - np.array(q_ref))) > 1e-6:
        ok = False; linea("    !! FALLO")

    linea("\n" + "=" * 72)
    linea("RESULTADO: " + ("TODO OK" if ok else "HAY FALLOS"))
    linea("=" * 72)
    return ok


def demo() -> None:
    """Muestra las 8 soluciones de un punto tipico del area de trabajo."""
    x, y, z, rx, ry, rz = -500.0, 100.0, 400.0, 180.0, 0.0, 0.0
    print(f"Pose objetivo (FAIRINO, mm/deg): "
          f"x={x} y={y} z={z} rx={rx} ry={ry} rz={rz}\n")
    sols = ik_fairino(x, y, z, rx, ry, rz,
                      q_actual_grados=[0, -90, 90, -90, -90, 0], todas=True)
    if not sols:
        print("Inalcanzable con los limites del URDF."); return
    print(f"{len(sols)} soluciones validas dentro de limites "
          f"(ordenadas por cercania a la posicion actual):\n")
    print(f"{'#':>2}  {'j1':>8} {'j2':>8} {'j3':>8} {'j4':>8} {'j5':>8} {'j6':>8}"
          f"  {'costo':>7} {'sigma':>6}  rama")
    for i, s in enumerate(sols):
        g = s.grados()
        print(f"{i:2d}  " + " ".join(f"{v:8.2f}" for v in g) +
              f"  {s.costo:7.3f} {s.manipulabilidad:6.3f}  {s.rama}")


# =============================================================================
# 7. CLI
# =============================================================================

def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Cinematica directa/inversa cerrada del FAIRINO FR5.")
    ap.add_argument("--selftest", action="store_true",
                    help="corre la bateria de verificacion")
    ap.add_argument("--demo", action="store_true",
                    help="muestra las 8 soluciones de una pose de ejemplo")
    ap.add_argument("--ik", metavar="x,y,z,rx,ry,rz",
                    help="resuelve una pose (mm y grados)")
    ap.add_argument("--fk", metavar="j1,...,j6",
                    help="cinematica directa desde juntas en grados")
    ap.add_argument("--q", metavar="j1,...,j6",
                    help="posicion actual del robot en grados (semilla)")
    ap.add_argument("--tcp", metavar="x,y,z,rx,ry,rz",
                    help="TCP leido del robot real (mm/grados), para --verificar")
    ap.add_argument("--verificar", action="store_true",
                    help="compara el modelo contra una lectura real (--q y --tcp)")
    ap.add_argument("--explicar", metavar="x,y,z,rx,ry,rz",
                    help="dice POR QUE una pose sale o no (mm y grados)")
    ap.add_argument("--d6", type=float, default=None,
                    help=f"offset del TCP sobre z6 en metros. Def. {D6_BRIDA} "
                         f"(brida). Usa {D6_TOOL_URDF} para incluir la "
                         f"herramienta del tool_Link.")
    args = ap.parse_args()

    params = FR5 if args.d6 is None else ParamsFR5(d6=args.d6)
    nums = lambda s: [float(v) for v in s.replace(" ", "").split(",")]

    if args.selftest:
        return 0 if selftest() else 1
    if args.demo:
        demo(); return 0
    if args.verificar:
        if not (args.q and args.tcp):
            ap.error("--verificar necesita --q y --tcp")
        r = verificar_contra_robot(nums(args.q), nums(args.tcp), params)
        for k, v in r.items():
            print(f"{k:24s}: {v}")
        return 0 if r["modelo_ok"] else 2
    if args.fk:
        print(", ".join(f"{v:.4f}" for v in fk_fairino(nums(args.fk), params)))
        return 0
    if args.explicar:
        x, y, z, rx, ry, rz = nums(args.explicar)
        print(explicar(pose_a_T(x / 1000.0, y / 1000.0, z / 1000.0,
                                math.radians(rx), math.radians(ry),
                                math.radians(rz)), params))
        return 0
    if args.ik:
        sols = ik_fairino(*nums(args.ik),
                          q_actual_grados=nums(args.q) if args.q else None,
                          params=params, todas=True)
        if not sols:
            print("INALCANZABLE"); return 3
        for i, s in enumerate(sols):
            print(f"{i:2d}  " + " ".join(f"{v:9.4f}" for v in s.grados()) +
                  f"   {s.rama}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
