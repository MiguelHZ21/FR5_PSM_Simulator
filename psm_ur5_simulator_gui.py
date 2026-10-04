#!/usr/bin/env python3
"""
psm_ur5_simulator_gui.py
=========================

Interfaz gráfica completa del sistema acoplado FR5 + PSM da Vinci EndoWrist.

Una sola ventana con dos paneles 3D lado a lado:
    - Izquierda: sistema completo (FR5 + trocar + PSM + cilindros + mesa).
    - Derecha:   vista zoom solo del PSM y la zona de trabajo.

Controles principales:
    - Configuración:
        Cantidad de cilindros (1-5)
        Modo de inicialización: HOME FR5 / Fulcro
        Coordenadas del fulcro (X, Y, Z) y dirección del shaft (visibles
        solo en modo Fulcro)
        Distancia trocar (TCP FR5 -> fulcro)
        Distancia total de la pinza
        Offset X, Y de colocación
        Seed (con botón "aleatoria")

    - Acciones:
        [Reset]
        [Generar escena]
        [Calcular trayectorias]
        [Exportar TXT]

    - Reproducción:
        [|<]  reset al wp 0



        [###] stop
        [||]  pause
        [>]   play
        Slider de velocidad (1-60 fps)
        Slider de progreso (waypoint 0..N)

    - Cámara:
        Sliders de elevación / azimut por panel
        Botones [+] / [-] de zoom por panel

Comportamiento:
    - Toda la configuración se bloquea durante la animación.
    - Si la IK del FR5 falla en algún waypoint, el TCP del PSM se dibuja
      en rojo y se anota en el log; la animación continúa.

Requisitos:
    pip install PyQt6 numpy matplotlib pillow \
                roboticstoolbox-python spatialmath-python
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

# --- Configurar matplotlib ANTES de pyplot --------------------------
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QTextEdit, QGroupBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QComboBox, QFileDialog,
    QMessageBox, QSplitter, QFrame, QToolButton, QSizePolicy, QScrollArea,
    QTabWidget,
)

# --- Módulos del proyecto -------------------------------------------
from psm_kinematics import Q_HOME_LAPARO, forward_kinematics, inverse_kinematics
from psm_scene import generate_random_cylinders
from psm_pick_place import plan_pick_and_place, replan_with_retries, resample_plan_and_fr5
from psm_ur5_kinematics import (
    DEFAULT_DIST_TROCAR, DEFAULT_DIST_TOTAL, DIST_TROCAR_RANGE,
    load_fr5, fr5_home_default,
    fr5_fk, fr5_ik,
    init_from_fr5_home, init_from_fulcrum,
    fr5_target_from_psm_state,
)
from psm_ur5_render import render_full_system, render_psm_only


# =============================================================================
# Canvas 3D
# =============================================================================

class Plot3DCanvas(FigureCanvas):
    """Canvas matplotlib 3D con un solo Axes y soporte de zoom por rueda."""

    zoom_changed = pyqtSignal(float)  # factor multiplicativo del zoom

    def __init__(self, parent=None, dark=False):
        bg = "#1a1a2e" if dark else "white"
        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor=bg)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="3d")
        # Conectar evento de scroll de la rueda del ratón
        self.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, event):
        """Zoom con la rueda del ratón."""
        if event.button == "up":
            factor = 0.9
        elif event.button == "down":
            factor = 1.1
        else:
            return
        self.zoom_changed.emit(factor)


# =============================================================================
# Ventana principal
# =============================================================================

class CoupledSimulatorMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador acoplado FR5 + PSM da Vinci EndoWrist")
        self.resize(1280, 800)

        # ----- Estado interno -----
        self.fr5_robot = None
        self.fr5_T_tool = None
        self.cylinders = []
        self.plan = None
        self.fr5_q_per_wp = None
        self.fr5_success_per_wp = None

        # Pose del FR5 fija al inicio
        self.q_fr5_init = None
        self.T_fr5_init = None
        self.p_fulcrum = None
        self.T_psm_base = None

        # Estado actual de animación
        self.current_idx = 0
        self.cyl_poses_local: list[np.ndarray] = []
        self.attached_idx: Optional[int] = None
        self.is_playing = False
        self.play_start_time = 0.0
        self.play_start_idx = 0

        # Estado de cámara por panel (rotado 180° en Z -> azim=-138°)
        self.cam_full = {"elev": 22.0, "azim": -138.0, "zoom_factor": 1.0}
        self.cam_psm  = {"elev": 22.0, "azim": -138.0, "zoom_factor": 1.0}

        # ----- Estado del modo manual (sliders X/Y/Z/RX/RY/RZ) -----
        # Pose de referencia del TCP del PSM (frame mundo). Por defecto se
        # captura cuando se inicializa el FR5 (pose del PSM en HOME laparoscópico).
        self.T_ref_tcp_world: Optional[np.ndarray] = None
        # Configuración articular del PSM correspondiente a la pose de referencia
        self.q_ref_psm: Optional[np.ndarray] = None
        # Configuración articular actual del PSM en modo manual
        self.q_manual_psm: Optional[np.ndarray] = None
        self.q_manual_fr5: Optional[np.ndarray] = None
        # Bandera para evitar bucles de recursión en los sliders
        self._manual_updating = False
        # Bandera para indicar fallo IK del PSM en pose manual
        self._manual_ik_failed = False

        # Timer de animación
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_step)

        self._build_ui()
        self._load_fr5_async()

    # =========================================================================
    # Construcción de UI
    # =========================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ----- Splitter principal: paneles 3D | controles -----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ===== Lado izquierdo: dos canvas 3D apilados horizontalmente =====
        left_widget = QWidget()
        left_layout = QHBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)

        # --- Panel sistema completo ---
        self.canvas_full = Plot3DCanvas(dark=True)
        self.canvas_full.zoom_changed.connect(
            lambda f: self._apply_zoom("full", f)
        )
        full_box = self._build_view_panel(
            "Sistema completo (FR5 + trocar + PSM)",
            self.canvas_full, prefix="full",
        )
        left_layout.addWidget(full_box, stretch=1)

        # --- Panel PSM solo ---
        self.canvas_psm = Plot3DCanvas(dark=False)
        self.canvas_psm.zoom_changed.connect(
            lambda f: self._apply_zoom("psm", f)
        )
        psm_box = self._build_view_panel(
            "Vista zoom del PSM y zona de trabajo",
            self.canvas_psm, prefix="psm",
        )
        left_layout.addWidget(psm_box, stretch=1)

        splitter.addWidget(left_widget)

        # ===== Lado derecho: panel con pestañas =====
        right_widget = QWidget()
        right_widget.setMinimumWidth(330)
        right_widget.setMaximumWidth(420)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Título compacto
        title = QLabel("Sistema acoplado FR5 + PSM")
        title.setFont(QFont("", 11, QFont.Weight.Bold))
        right_layout.addWidget(title)

        # ----- Pestañas -----
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_configuracion(), "Configuración")
        self.tabs.addTab(self._build_tab_reproduccion(), "Reproducción")
        self.tabs.addTab(self._build_tab_pinza_manual(), "Pinza manual")
        self.tabs.addTab(self._build_tab_fr5_manual(), "FR5 manual")
        self.tabs.addTab(self._build_tab_camara(), "Cámara")
        right_layout.addWidget(self.tabs, stretch=2)

        # ----- Log compartido (siempre visible debajo de las pestañas) -----
        right_layout.addWidget(self._build_log(), stretch=1)

        # ----- Salir -----
        btn_quit = QPushButton("Salir")
        btn_quit.clicked.connect(self.close)
        right_layout.addWidget(btn_quit)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        self._log("Cargando modelo FR5...")

    def _build_view_panel(self, title: str, canvas: FigureCanvas, prefix: str):
        """Construye un GroupBox con un canvas 3D y botones de zoom rápido."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(canvas, stretch=1)

        # Botones de zoom en barra horizontal compacta (acceso rápido)
        zoom_row = QHBoxLayout()
        zoom_row.setContentsMargins(0, 0, 0, 0)
        btn_zoom_in = QToolButton()
        btn_zoom_in.setText("Zoom +")
        btn_zoom_in.setToolTip("Acercar la vista")
        btn_zoom_in.clicked.connect(
            lambda _=False, p=prefix: self._apply_zoom(p, 0.85)
        )
        btn_zoom_out = QToolButton()
        btn_zoom_out.setText("Zoom -")
        btn_zoom_out.setToolTip("Alejar la vista")
        btn_zoom_out.clicked.connect(
            lambda _=False, p=prefix: self._apply_zoom(p, 1.18)
        )
        btn_zoom_in.setMinimumHeight(22)
        btn_zoom_out.setMinimumHeight(22)
        zoom_row.addWidget(btn_zoom_in)
        zoom_row.addWidget(btn_zoom_out)
        zoom_row.addStretch(1)
        layout.addLayout(zoom_row)
        return box

    # =========================================================================
    # Pestañas del panel derecho
    # =========================================================================
    def _build_tab_configuracion(self) -> QWidget:
        """Pestaña 1: configuración de escena + FR5 + acciones."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_scene_config())
        layout.addWidget(self._build_ur5_config())
        layout.addWidget(self._build_action_buttons())
        layout.addStretch(1)
        return tab

    def _build_tab_reproduccion(self) -> QWidget:
        """Pestaña 2: controles de animación."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_playback_controls())
        layout.addStretch(1)
        return tab

    def _build_tab_pinza_manual(self) -> QWidget:
        """Pestaña 3: sliders articulares (shaft, wrist, jaw_dx, jaw_sx) para la pinza EndoWrist."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        # ----- Encabezado: estado del modo manual -----
        gb_estado = QGroupBox("Estado del modo manual (EndoWrist)")
        est_layout = QVBoxLayout(gb_estado)
        self.lbl_manual_estado = QLabel(
            "Control articular de la pinza EndoWrist (4 GDL).\n"
            "Mueve únicamente la pinza (shaft, wrist, jaw_dx, jaw_sx).\n"
            "El robot FR5 y las 3 primeras articulaciones del PSM permanecen fijos."
        )
        self.lbl_manual_estado.setStyleSheet("color: #555;")
        self.lbl_manual_estado.setWordWrap(True)
        est_layout.addWidget(self.lbl_manual_estado)
        layout.addWidget(gb_estado)

        # ----- Botón de control -----
        btn_row = QHBoxLayout()
        self.btn_resetear_pinza = QPushButton("Resetear pinza a 0°")
        self.btn_resetear_pinza.clicked.connect(self.on_resetear_pinza)
        btn_row.addWidget(self.btn_resetear_pinza)
        layout.addLayout(btn_row)

        # ----- Sliders Articulares -----
        gb_art = QGroupBox("Coordenadas Articulares de la Pinza [°]")
        f_art = QFormLayout(gb_art)
        self.sliders_endowrist = {}

        joint_defs = [
            ("shaft",  "Shaft (Tool Roll)",       -180.0, 180.0, 1.0, 0, " °"),
            ("wrist",  "Wrist (Wrist Pitch)",     -70.0,   70.0, 1.0, 0, " °"),
            ("jaw_dx", "Jaw DX (Mordaza Der.)",   -60.0,   60.0, 1.0, 0, " °"),
            ("jaw_sx", "Jaw SX (Mordaza Izq.)",   -60.0,   60.0, 1.0, 0, " °"),
        ]

        for key, label_name, vmin, vmax, step, dec, suff in joint_defs:
            slider, spin, row = self._make_slider_with_spin(
                vmin=vmin, vmax=vmax, step=step, decimals=dec,
                suffix=suff, lo_label=key,
            )
            f_art.addRow(f"{label_name}:", row)
            self.sliders_endowrist[key] = (slider, spin)
            slider.valueChanged.connect(
                lambda v, k=key: self._on_endowrist_slider_changed(k)
            )
            spin.valueChanged.connect(
                lambda v, k=key: self._on_endowrist_spin_changed(k)
            )

        layout.addWidget(gb_art)
        layout.addStretch(1)
        return tab

    def _make_slider_with_spin(self, vmin, vmax, step, decimals, suffix, lo_label):
        """Crea un slider + spinbox sincronizados. Devuelve (slider, spin, layout)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        # Internal range: scaled by 1/step for integer ticks
        scale = int(round(1.0 / step))
        slider.setRange(int(vmin * scale), int(vmax * scale))
        slider.setValue(0)
        slider.setMinimumWidth(120)
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(0.0)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(70)
        row.addWidget(slider, stretch=2)
        row.addWidget(spin, stretch=0)
        # Asociar metadata para escala
        slider._scale_factor = scale
        return slider, spin, row

    def _build_tab_camara(self) -> QWidget:
        """Pestaña 4: sliders de cámara para los dos paneles 3D."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        for prefix, title in (("full", "Panel sistema completo"),
                              ("psm",  "Panel PSM solo")):
            gb = QGroupBox(title)
            form = QFormLayout(gb)

            slider_elev = QSlider(Qt.Orientation.Horizontal)
            slider_elev.setRange(-90, 90); slider_elev.setValue(22)
            slider_elev.valueChanged.connect(
                lambda v, p=prefix: self._on_cam_change(p, "elev", v)
            )
            slider_azim = QSlider(Qt.Orientation.Horizontal)
            slider_azim.setRange(-180, 180); slider_azim.setValue(-138)
            slider_azim.valueChanged.connect(
                lambda v, p=prefix: self._on_cam_change(p, "azim", v)
            )
            form.addRow("Elev:", slider_elev)
            form.addRow("Azim:", slider_azim)

            # Botones de zoom
            zoom_row = QHBoxLayout()
            btn_in = QPushButton("Zoom +")
            btn_in.clicked.connect(
                lambda _=False, p=prefix: self._apply_zoom(p, 0.85)
            )
            btn_out = QPushButton("Zoom -")
            btn_out.clicked.connect(
                lambda _=False, p=prefix: self._apply_zoom(p, 1.18)
            )
            zoom_row.addWidget(btn_in)
            zoom_row.addWidget(btn_out)
            form.addRow("Zoom:", zoom_row)

            setattr(self, f"slider_elev_{prefix}", slider_elev)
            setattr(self, f"slider_azim_{prefix}", slider_azim)

            layout.addWidget(gb)

        layout.addStretch(1)
        return tab

    def _build_tab_fr5_manual(self) -> QWidget:
        """Pestaña 5: sliders articulares (q1 a q6) para el FR5."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        gb_estado = QGroupBox("Estado del FR5")
        est_layout = QVBoxLayout(gb_estado)
        lbl = QLabel(
            "Modo activo cuando no hay plan calculado.\n"
            "Mueve las 6 articulaciones del FR5 directamente (±180°)."
        )
        lbl.setStyleSheet("color: #555;")
        lbl.setWordWrap(True)
        est_layout.addWidget(lbl)
        layout.addWidget(gb_estado)
        
        btn_row = QHBoxLayout()
        self.btn_resetear_fr5 = QPushButton("Resetear FR5 a Home")
        self.btn_resetear_fr5.clicked.connect(self.on_resetear_fr5)
        btn_row.addWidget(self.btn_resetear_fr5)
        layout.addLayout(btn_row)

        gb_art = QGroupBox("Coordenadas Articulares [°]")
        f_art = QFormLayout(gb_art)
        self.sliders_fr5 = []
        limits_deg = [
            (-175.0, 175.0),  # q1 (0° real en cinemática)
            (-265.0, 85.0),   # q2
            (-162.0, 162.0),  # q3
            (-265.0, 85.0),   # q4
            (-175.0, 175.0),  # q5
            (-175.0, 175.0),  # q6
        ]
        for i in range(6):
            vmin, vmax = limits_deg[i]
            slider, spin, row = self._make_slider_with_spin(
                vmin=vmin, vmax=vmax, step=1.0, decimals=1,
                suffix=" °", lo_label=f"q{i+1}",
            )
            f_art.addRow(f"q{i+1}:", row)
            self.sliders_fr5.append((slider, spin))
            slider.valueChanged.connect(
                lambda v, idx=i: self._on_fr5_slider_changed(idx)
            )
            spin.valueChanged.connect(
                lambda v, idx=i: self._on_fr5_spin_changed(idx)
            )
        layout.addWidget(gb_art)
        layout.addStretch(1)
        return tab


    def _build_scene_config(self):
        gb = QGroupBox("Configuración de la escena")
        form = QFormLayout(gb)

        self.spin_n_cyl = QSpinBox()
        self.spin_n_cyl.setRange(1, 5); self.spin_n_cyl.setValue(3)
        form.addRow("Cilindros:", self.spin_n_cyl)

        seed_row = QHBoxLayout()
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999_999_999)
        self.spin_seed.setValue(42)
        btn_random_seed = QPushButton("Aleatoria")
        btn_random_seed.setFixedWidth(80)
        btn_random_seed.clicked.connect(self._on_random_seed)
        seed_row.addWidget(self.spin_seed, stretch=1)
        seed_row.addWidget(btn_random_seed)
        form.addRow("Seed:", seed_row)

        self.spin_offset_x = QDoubleSpinBox()
        self.spin_offset_x.setRange(-0.05, 0.05)
        self.spin_offset_x.setSingleStep(0.005)
        self.spin_offset_x.setDecimals(3)
        self.spin_offset_x.setValue(0.020)
        self.spin_offset_x.setSuffix(" m")
        form.addRow("Offset X:", self.spin_offset_x)

        self.spin_offset_y = QDoubleSpinBox()
        self.spin_offset_y.setRange(-0.05, 0.05)
        self.spin_offset_y.setSingleStep(0.005)
        self.spin_offset_y.setDecimals(3)
        self.spin_offset_y.setValue(0.000)
        self.spin_offset_y.setSuffix(" m")
        form.addRow("Offset Y:", self.spin_offset_y)

        self.spin_n_points = QSpinBox()
        self.spin_n_points.setRange(100, 10000)
        self.spin_n_points.setSingleStep(100)
        self.spin_n_points.setValue(2000)
        form.addRow("Puntos trayectoria:", self.spin_n_points)

        return gb

    def _build_ur5_config(self):
        gb = QGroupBox("Configuración FR5 + trocar")
        layout = QVBoxLayout(gb)

        # Modo de inicialización
        mode_row = QFormLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["HOME del FR5", "Fulcro especificado"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addRow("Modo init:", self.combo_mode)
        layout.addLayout(mode_row)

        # Container del modo Fulcro (oculto por defecto)
        self.fulcrum_widget = QWidget()
        fulc_layout = QFormLayout(self.fulcrum_widget)
        fulc_layout.setContentsMargins(0, 0, 0, 0)

        # Coordenadas del fulcro
        fulc_xyz = QHBoxLayout()
        self.spin_fulc_x = QDoubleSpinBox()
        self.spin_fulc_x.setRange(-1.0, 1.0); self.spin_fulc_x.setSingleStep(0.01)
        self.spin_fulc_x.setDecimals(3); self.spin_fulc_x.setValue(0.55)
        self.spin_fulc_y = QDoubleSpinBox()
        self.spin_fulc_y.setRange(-1.0, 1.0); self.spin_fulc_y.setSingleStep(0.01)
        self.spin_fulc_y.setDecimals(3); self.spin_fulc_y.setValue(-0.10)
        self.spin_fulc_z = QDoubleSpinBox()
        self.spin_fulc_z.setRange(0.0, 1.5); self.spin_fulc_z.setSingleStep(0.01)
        self.spin_fulc_z.setDecimals(3); self.spin_fulc_z.setValue(0.45)
        for s in (self.spin_fulc_x, self.spin_fulc_y, self.spin_fulc_z):
            s.setSuffix(" m")
            fulc_xyz.addWidget(s)
        fulc_layout.addRow("Fulcro:", fulc_xyz)

        # Dirección del shaft
        sh_xyz = QHBoxLayout()
        self.spin_shaft_x = QDoubleSpinBox()
        self.spin_shaft_x.setRange(-1.0, 1.0); self.spin_shaft_x.setSingleStep(0.05)
        self.spin_shaft_x.setDecimals(3); self.spin_shaft_x.setValue(0.5)
        self.spin_shaft_y = QDoubleSpinBox()
        self.spin_shaft_y.setRange(-1.0, 1.0); self.spin_shaft_y.setSingleStep(0.05)
        self.spin_shaft_y.setDecimals(3); self.spin_shaft_y.setValue(0.0)
        self.spin_shaft_z = QDoubleSpinBox()
        self.spin_shaft_z.setRange(-1.0, 1.0); self.spin_shaft_z.setSingleStep(0.05)
        self.spin_shaft_z.setDecimals(3); self.spin_shaft_z.setValue(-0.866)
        for s in (self.spin_shaft_x, self.spin_shaft_y, self.spin_shaft_z):
            sh_xyz.addWidget(s)
        fulc_layout.addRow("Dir shaft:", sh_xyz)

        layout.addWidget(self.fulcrum_widget)
        self.fulcrum_widget.setVisible(False)

        # Distancia trocar y total
        dist_form = QFormLayout()
        self.spin_dist_trocar = QDoubleSpinBox()
        self.spin_dist_trocar.setRange(*DIST_TROCAR_RANGE)
        self.spin_dist_trocar.setSingleStep(0.01)
        self.spin_dist_trocar.setDecimals(3)
        self.spin_dist_trocar.setValue(DEFAULT_DIST_TROCAR)
        self.spin_dist_trocar.setSuffix(" m")
        dist_form.addRow("Dist. trocar:", self.spin_dist_trocar)

        self.spin_dist_total = QDoubleSpinBox()
        self.spin_dist_total.setRange(0.20, 0.60)
        self.spin_dist_total.setSingleStep(0.01)
        self.spin_dist_total.setDecimals(3)
        self.spin_dist_total.setValue(DEFAULT_DIST_TOTAL)
        self.spin_dist_total.setSuffix(" m")
        dist_form.addRow("Long. total:", self.spin_dist_total)

        layout.addLayout(dist_form)
        return gb

    def _build_action_buttons(self):
        gb = QGroupBox("Acciones")
        layout = QVBoxLayout(gb)

        self.btn_reset = QPushButton("Reset (HOME, escena vacía)")
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reset.setStyleSheet("padding: 6px;")

        self.btn_generate = QPushButton("1. Generar escena")
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_generate.setStyleSheet("padding: 6px; font-weight: bold;")

        self.btn_plan = QPushButton("2. Calcular trayectorias")
        self.btn_plan.clicked.connect(self.on_plan)
        self.btn_plan.setStyleSheet("padding: 6px; font-weight: bold;")
        self.btn_plan.setEnabled(False)

        self.btn_export = QPushButton("Exportar TXT (articulares + cartesianas)")
        self.btn_export.clicked.connect(self.on_export)
        self.btn_export.setStyleSheet("padding: 6px;")
        self.btn_export.setEnabled(False)

        for b in (self.btn_reset, self.btn_generate, self.btn_plan, self.btn_export):
            layout.addWidget(b)
        return gb

    def _build_playback_controls(self):
        gb = QGroupBox("Reproducción")
        layout = QVBoxLayout(gb)

        # Botones transporte
        btn_row = QHBoxLayout()
        self.btn_rewind = QPushButton("⏮")
        self.btn_stop   = QPushButton("⏹")
        self.btn_pause  = QPushButton("⏸")
        self.btn_play   = QPushButton("▶")
        for b in (self.btn_rewind, self.btn_stop, self.btn_pause, self.btn_play):
            b.setMinimumHeight(32)
            b.setStyleSheet("font-size: 14pt; padding: 4px;")
            btn_row.addWidget(b)
        self.btn_rewind.clicked.connect(self.on_rewind)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_play.clicked.connect(self.on_play)
        layout.addLayout(btn_row)

        # Velocidad
        speed_form = QFormLayout()
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 120); self.slider_speed.setValue(30)
        self.lbl_speed = QLabel("30 fps")
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        speed_form.addRow("Velocidad:", self.slider_speed)
        speed_form.addRow("", self.lbl_speed)

        # Progreso
        self.slider_progress = QSlider(Qt.Orientation.Horizontal)
        self.slider_progress.setRange(0, 0)
        self.slider_progress.setValue(0)
        self.slider_progress.sliderMoved.connect(self._on_progress_seek)
        self.lbl_progress = QLabel("WP: 0 / 0")
        speed_form.addRow("Progreso:", self.slider_progress)
        speed_form.addRow("", self.lbl_progress)

        layout.addLayout(speed_form)

        # Por defecto deshabilitar todo hasta que haya plan
        self._enable_playback(False)
        return gb

    def _build_log(self):
        gb = QGroupBox("Log")
        layout = QVBoxLayout(gb)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 8))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        layout.addWidget(self.log_text)
        return gb

    # =========================================================================
    # Helpers
    # =========================================================================
    def _log(self, msg: str, color: str = "#d4d4d4"):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(
            f'<span style="color:#888">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>'
        )

    def _load_fr5_async(self):
        """Carga el FR5 (puede tardar 1-2 s) usando un timer single-shot."""
        QTimer.singleShot(50, self._do_load_fr5)

    def _do_load_fr5(self):
        try:
            self.fr5_robot, self.fr5_T_tool = load_fr5()
            self._log("Modelo FR5 cargado.")
            self._reinit_ur5_pose()
        except Exception as e:
            self._log(f"ERROR cargando FR5: {e}", color="#f55")
            QMessageBox.critical(self, "Error", f"No se pudo cargar FR5:\n{e}")
            return
        # Capturar pose de referencia inicial para el modo manual
        self._capture_reference_pose()
        self._render_initial()

    def _reinit_ur5_pose(self):
        """Calcula FR5/fulcro/T_psm_base según el modo configurado."""
        try:
            dist_trocar = self.spin_dist_trocar.value()
            if self.combo_mode.currentIndex() == 0:
                # HOME del FR5
                self.q_fr5_init, self.T_fr5_init, self.p_fulcrum, self.T_psm_base = \
                    init_from_fr5_home(self.fr5_robot, self.fr5_T_tool, dist_trocar=dist_trocar)
                self._log(f"Init HOME FR5: fulcro = "
                          f"{self.p_fulcrum.round(3).tolist()} m")
            else:
                # Fulcro especificado
                p_fulc = np.array([self.spin_fulc_x.value(),
                                   self.spin_fulc_y.value(),
                                   self.spin_fulc_z.value()])
                n = np.array([self.spin_shaft_x.value(),
                              self.spin_shaft_y.value(),
                              self.spin_shaft_z.value()])
                if np.linalg.norm(n) < 1e-6:
                    raise ValueError("La dirección del shaft no puede ser nula.")
                n = n / np.linalg.norm(n)
                (self.q_fr5_init, self.T_fr5_init, self.p_fulcrum,
                 self.T_psm_base, ok, err) = init_from_fulcrum(
                    self.fr5_robot, self.fr5_T_tool, p_fulc, n, dist_trocar=dist_trocar,
                )
                if not ok:
                    raise RuntimeError(
                        f"IK del FR5 no alcanza el fulcro deseado "
                        f"(error {err:.2f} mm). Pruebe otra pose."
                    )
                self._log(f"Init Fulcro: fulcro = "
                          f"{self.p_fulcrum.round(3).tolist()} m, IK err = {err:.4f} mm")
        except Exception as e:
            self._log(f"ERROR inicializando FR5: {e}", color="#f55")
            return False
        return True

    # =========================================================================
    # Render
    # =========================================================================
    def _render_initial(self):
        """Render inicial: solo FR5 en HOME, sin cilindros, sin plan."""
        if self.fr5_robot is None:
            return
        info_full = "HOME FR5\nSin cilindros"
        info_psm = "HOME PSM laparoscopico\nSin cilindros"

        render_full_system(
            self.canvas_full.ax, self.fr5_robot,
            self.q_fr5_init, self.T_fr5_init,
            self.spin_dist_trocar.value(),
            self.p_fulcrum,
            Q_HOME_LAPARO, self.T_psm_base,
            [], [], None,
            info_text=info_full,
            elev=self.cam_full["elev"], azim=self.cam_full["azim"],
        )
        render_psm_only(
            self.canvas_psm.ax,
            Q_HOME_LAPARO, self.T_psm_base, self.p_fulcrum,
            [], [], None,
            info_text=info_psm,
            elev=self.cam_psm["elev"], azim=self.cam_psm["azim"],
            zoom_factor=self.cam_psm["zoom_factor"],
        )
        self._apply_full_zoom_factor()
        self.canvas_full.draw_idle()
        self.canvas_psm.draw_idle()

    def _render_current_state(self):
        """Render del estado actual (cilindros + waypoint si hay plan)."""
        if self.fr5_robot is None or self.plan is None:
            return

        idx = self.current_idx
        q_psm = self.plan.Q[idx]
        q_fr5 = self.fr5_q_per_wp[idx]
        ik_failed = not self.fr5_success_per_wp[idx]
        wp = self.plan.waypoints[idx]
        T_fr5 = fr5_fk(self.fr5_robot, self.fr5_T_tool, q_fr5)

        info = (
            f"WP {idx+1}/{len(self.plan.waypoints)}\n"
            f"Fase: {wp.phase}  |  IK FR5: {'OK' if not ik_failed else 'FAIL'}\n"
            f"Cil agarrado: "
            f"{'C'+str(wp.cylinder_idx) if wp.cylinder_attached else '—'}"
        )

        render_full_system(
            self.canvas_full.ax, self.fr5_robot,
            q_fr5, T_fr5, self.spin_dist_trocar.value(), self.p_fulcrum,
            q_psm, self.T_psm_base,
            self.cylinders, self.cyl_poses_local, self.attached_idx,
            ik_failed=ik_failed, info_text=info,
            elev=self.cam_full["elev"], azim=self.cam_full["azim"],
        )
        render_psm_only(
            self.canvas_psm.ax,
            q_psm, self.T_psm_base, self.p_fulcrum,
            self.cylinders, self.cyl_poses_local, self.attached_idx,
            ik_failed=ik_failed, info_text=info,
            elev=self.cam_psm["elev"], azim=self.cam_psm["azim"],
            zoom_factor=self.cam_psm["zoom_factor"],
        )
        self._apply_full_zoom_factor()
        self.canvas_full.draw_idle()
        self.canvas_psm.draw_idle()

    def _apply_full_zoom_factor(self):
        """Aplica zoom_factor al panel del sistema completo (cambia rangos)."""
        zf = self.cam_full["zoom_factor"]
        # Centro y span por defecto
        cx, cy, cz = -0.40, -0.025, 0.50
        sx, sy, sz = 0.40, 0.525, 0.50  # half-spans (X: -0.8 a 0)
        sx /= zf; sy /= zf; sz /= zf
        self.canvas_full.ax.set_xlim(cx - sx, cx + sx)
        self.canvas_full.ax.set_ylim(cy - sy, cy + sy)
        self.canvas_full.ax.set_zlim(max(0.0, cz - sz), cz + sz)

    # =========================================================================
    # Cámara: callbacks
    # =========================================================================
    def _on_cam_change(self, prefix: str, kind: str, value: int):
        cam = self.cam_full if prefix == "full" else self.cam_psm
        cam[kind] = float(value)
        # Re-render rápido (sin recalcular nada, solo cambia cámara)
        if self.plan is None:
            self._render_initial()
        else:
            self._render_current_state()

    def _apply_zoom(self, prefix: str, factor: float):
        cam = self.cam_full if prefix == "full" else self.cam_psm
        # Limitar el factor de zoom acumulado
        new_zf = cam["zoom_factor"] / factor   # factor<1 = zoom in
        new_zf = max(0.3, min(5.0, new_zf))
        cam["zoom_factor"] = new_zf
        if self.plan is None:
            self._render_initial()
        else:
            self._render_current_state()

    # =========================================================================
    # Slot: cambio de modo init
    # =========================================================================
    def _on_mode_changed(self, idx):
        self.fulcrum_widget.setVisible(idx == 1)
        # Forzar recálculo del FR5 (descartando escena/plan)
        self._reset_state()
        if self._reinit_ur5_pose():
            self._log(f"Modo cambiado a "
                      f"'{['HOME FR5', 'Fulcro especificado'][idx]}'. "
                      f"Pulse [Generar escena].")
            self._render_initial()

    def _on_random_seed(self):
        self.spin_seed.setValue(np.random.randint(0, 999_999_999))

    def _on_progress_seek(self, value):
        if self.plan is None or self.is_playing:
            return
        # Saltar al waypoint indicado
        self._set_current_idx(value)
        self._render_current_state()

    # =========================================================================
    # Acciones principales
    # =========================================================================
    def _reset_state(self):
        """Borra escena/plan y vuelve a HOME."""
        self.cylinders = []
        self.plan = None
        self.fr5_q_per_wp = None
        self.fr5_success_per_wp = None
        self.current_idx = 0
        self.cyl_poses_local = []
        self.attached_idx = None
        self.is_playing = False
        self.anim_timer.stop()
        self.btn_plan.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._enable_playback(False)
        self.slider_progress.setRange(0, 0)
        self.slider_progress.setValue(0)
        self.lbl_progress.setText("WP: 0 / 0")

    def on_reset(self):
        self._log("Reset: HOME, escena vacía.")
        self._reset_state()
        self._reinit_ur5_pose()
        # Re-capturar la pose de referencia del modo manual
        self._capture_reference_pose()
        self._render_initial()
        self._set_config_enabled(True)

    def on_generate(self):
        if self.fr5_robot is None:
            self._log("FR5 aún no cargado.", color="#fa6")
            return
        # Recalcular FR5 init (por si cambiaron parámetros)
        if not self._reinit_ur5_pose():
            return

        n_cyl = self.spin_n_cyl.value()
        seed = self.spin_seed.value()
        self._log(f"Generando {n_cyl} cilindros (seed={seed})...")

        try:
            rng = np.random.default_rng(seed)
            self.cylinders = generate_random_cylinders(n_cyl, rng=rng)
        except Exception as e:
            self._log(f"ERROR generando cilindros: {e}", color="#f55")
            return

        self.cyl_poses_local = [c.pose_homogeneous() for c in self.cylinders]
        self.plan = None
        self.fr5_q_per_wp = None
        self.attached_idx = None

        for c in self.cylinders:
            kind = "vertical" if c.is_vertical else "horizontal"
            p = c.position
            self._log(f"  {c.label} ({kind}): "
                      f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m",
                      color="#9cf")

        self.btn_plan.setEnabled(True)
        self.btn_export.setEnabled(False)
        self._enable_playback(False)
        self._render_initial_with_cylinders()

    def _render_initial_with_cylinders(self):
        """Render inicial pero ya con cilindros generados."""
        if self.fr5_robot is None:
            return
        info_full = (f"HOME FR5\n{len(self.cylinders)} cilindros\n"
                     f"Pulse [Calcular trayectorias]")
        info_psm = f"HOME PSM\n{len(self.cylinders)} cilindros generados"

        render_full_system(
            self.canvas_full.ax, self.fr5_robot,
            self.q_fr5_init, self.T_fr5_init,
            self.spin_dist_trocar.value(), self.p_fulcrum,
            Q_HOME_LAPARO, self.T_psm_base,
            self.cylinders, self.cyl_poses_local, None,
            info_text=info_full,
            elev=self.cam_full["elev"], azim=self.cam_full["azim"],
        )
        render_psm_only(
            self.canvas_psm.ax,
            Q_HOME_LAPARO, self.T_psm_base, self.p_fulcrum,
            self.cylinders, self.cyl_poses_local, None,
            info_text=info_psm,
            elev=self.cam_psm["elev"], azim=self.cam_psm["azim"],
            zoom_factor=self.cam_psm["zoom_factor"],
        )
        self._apply_full_zoom_factor()
        self.canvas_full.draw_idle()
        self.canvas_psm.draw_idle()

    def on_plan(self):
        if not self.cylinders:
            self._log("No hay cilindros.", color="#f55"); return

        self._log("Calculando plan PSM...")
        offset = np.array([self.spin_offset_x.value(),
                           self.spin_offset_y.value(), 0.0])

        plan, success, msg = plan_pick_and_place(
            self.cylinders, place_offset=offset, q_home=Q_HOME_LAPARO,
        )
        self._log(msg, color=("#9f9" if success else "#fa6"))

        if not success:
            self._log("Regenerando escena hasta encontrar alcanzable...",
                      color="#fa6")
            seed_initial = np.random.randint(1, 10_000_000)
            cyls, plan, attempts, msg2 = replan_with_retries(
                rng_seed_initial=seed_initial,
                n_cyl=len(self.cylinders),
                place_offset=offset, q_home=Q_HOME_LAPARO,
                max_retries=20,
            )
            if plan is None:
                self._log("FALLÓ tras 20 reintentos.", color="#f55")
                return
            self.cylinders = cyls
            self.cyl_poses_local = [c.pose_homogeneous() for c in cyls]
            self._log(f"Escena alcanzable encontrada en intento {attempts}.",
                      color="#9f9")

        self.plan = plan

        # Calcular IK del FR5 para cada waypoint
        self._log("Resolviendo IK del FR5 por waypoint...")
        n_wp = len(plan.waypoints)
        self.fr5_q_per_wp = np.zeros((n_wp, 6))
        self.fr5_success_per_wp = np.zeros(n_wp, dtype=bool)
        seed = self.q_fr5_init.copy()
        n_failed = 0
        dist_t = self.spin_dist_trocar.value()
        for i in range(n_wp):
            T_target = fr5_target_from_psm_state(
                plan.Q[i], self.p_fulcrum, self.T_psm_base,
                dist_trocar=self.spin_dist_trocar.value()
            )
            q_sol, ok, err = fr5_ik(self.fr5_robot, self.fr5_T_tool, T_target, seed)
            if ok:
                self.fr5_q_per_wp[i] = q_sol
                self.fr5_success_per_wp[i] = True
                seed = q_sol
            else:
                self.fr5_q_per_wp[i] = seed
                self.fr5_success_per_wp[i] = False
                n_failed += 1

        n_ok = int(self.fr5_success_per_wp.sum())
        if n_failed > 0:
            self._log(f"IK FR5 base: {n_ok}/{n_wp} OK, {n_failed} fallos.",
                      color="#fa6")
        else:
            self._log(f"IK FR5 base: {n_ok}/{n_wp} OK (100%).", color="#9f9")

        # Interpolación/Re-muestreo de alta densidad (2000 puntos para máxima fluidez)
        target_pts = self.spin_n_points.value()
        if target_pts != n_wp:
            self._log(f"Re-muestreando trayectoria a {target_pts} puntos para máxima fluidez...", color="#4fc3f7")
            self.plan, self.fr5_q_per_wp = resample_plan_and_fr5(
                plan, self.fr5_q_per_wp, target_n_points=target_pts
            )
            n_wp = len(self.plan.waypoints)
            self.fr5_success_per_wp = np.ones(n_wp, dtype=bool)

        # Habilitar reproducción
        self.current_idx = 0
        self.cyl_poses_local = [c.pose_homogeneous() for c in self.cylinders]
        self.attached_idx = None
        self.btn_export.setEnabled(True)
        self._enable_playback(True)
        self.slider_progress.setRange(0, n_wp - 1)
        self.slider_progress.setValue(0)
        self.lbl_progress.setText(f"WP: 1 / {n_wp}")
        self._render_current_state()
        self._log("Plan listo. Pulse [▶] para ejecutar.")

    def on_export(self):
        if self.plan is None:
            return
        # Diálogo de carpeta
        folder = QFileDialog.getExistingDirectory(
            self, "Carpeta destino para los TXT",
            os.path.abspath("./resultados_gui_ur5"),
        )
        if not folder:
            return
        try:
            self._do_export(folder)
            self._log(f"TXT exportados a: {folder}", color="#9f9")
        except Exception as e:
            self._log(f"ERROR exportando: {e}", color="#f55")

    def _do_export(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        from psm_ur5_kinematics import R_a_rpy

        # Archivo ArticularPoints (y CartesianPoints equivalente)
        path_art = os.path.join(folder, f"ArticularPoints_{ts}.txt")
        path_cart = os.path.join(folder, f"CartesianPoints_{ts}.txt")
        
        with open(path_art, "w") as fa, open(path_cart, "w") as fc:
            fa.write("articular\n")
            fc.write("cartesiano\n")
            
            for i, wp in enumerate(self.plan.waypoints):
                q = self.plan.Q[i]
                from psm_scene import GRIPPER_OPEN_RAD
                shaft_deg = np.rad2deg(q[3])
                wrist_deg = np.rad2deg(q[4])
                # Convención Unity: jaw_dx = +35.0° (+60° max), jaw_sx = -35.0° (-60° max) cuando está abierta
                ratio = q[6] / GRIPPER_OPEN_RAD if abs(GRIPPER_OPEN_RAD) > 1e-6 else 0.0
                jaw_dx_deg = 35.0 * ratio
                jaw_sx_deg = -35.0 * ratio
                
                # Pose del FR5
                T_u = fr5_fk(self.fr5_robot, self.fr5_T_tool, self.fr5_q_per_wp[i])
                x, y, z = T_u[:3, 3] * 1000.0  # en mm
                rx, ry, rz = R_a_rpy(T_u[:3, :3])
                rx, ry, rz = np.rad2deg([rx, ry, rz])
                
                speed = 10.0
                delay = 0.0
                
                line = (f"{x:.2f},{y:.2f},{z:.2f},{rx:.2f},{ry:.2f},{rz:.2f},"
                        f"{speed:.1f},{delay:.1f},"
                        f"{shaft_deg:.6f},{wrist_deg:.6f},{jaw_dx_deg:.6f},{jaw_sx_deg:.6f}\n")
                
                fa.write(line)
                fc.write(line)

    # =========================================================================
    # Reproducción
    # =========================================================================
    def _enable_playback(self, enabled: bool):
        for b in (self.btn_rewind, self.btn_stop, self.btn_pause, self.btn_play):
            b.setEnabled(enabled)
        self.slider_progress.setEnabled(enabled)

    def _set_config_enabled(self, enabled: bool):
        """Bloquea/desbloquea la configuración (durante reproducción)."""
        widgets = [
            self.spin_n_cyl, self.spin_seed, self.spin_offset_x, self.spin_offset_y,
            self.combo_mode, self.spin_fulc_x, self.spin_fulc_y, self.spin_fulc_z,
            self.spin_shaft_x, self.spin_shaft_y, self.spin_shaft_z,
            self.spin_dist_trocar, self.spin_dist_total,
            self.btn_reset, self.btn_generate, self.btn_plan,
        ]
        for w in widgets:
            w.setEnabled(enabled)
        # Modo manual: se bloquea cuando hay un plan calculado
        self._set_manual_enabled(enabled and self.plan is None)

    def _set_manual_enabled(self, enabled: bool):
        """Habilita/deshabilita los sliders de pinza manual."""
        manual_widgets = [self.btn_resetear_pinza]
        if hasattr(self, 'sliders_endowrist'):
            for slider, spin in self.sliders_endowrist.values():
                manual_widgets.append(slider); manual_widgets.append(spin)
        for w in manual_widgets:
            w.setEnabled(enabled)

    def _set_current_idx(self, idx: int):
        idx = max(0, min(len(self.plan.waypoints) - 1, idx))
        # Reconstruir el estado de los cilindros desde el inicio hasta idx
        self.cyl_poses_local = [c.pose_homogeneous() for c in self.cylinders]
        self.attached_idx = None
        for j in range(idx + 1):
            wp_j = self.plan.waypoints[j]
            if wp_j.cylinder_attached and wp_j.grasp_T_rel is not None:
                T_tcp_local = forward_kinematics(self.plan.Q[j])
                self.cyl_poses_local[wp_j.cylinder_idx] = T_tcp_local @ wp_j.grasp_T_rel
                self.attached_idx = wp_j.cylinder_idx
            elif j > 0:
                prev = self.plan.waypoints[j - 1]
                if (prev.cylinder_attached and not wp_j.cylinder_attached
                        and wp_j.cylinder_idx == prev.cylinder_idx):
                    self.cyl_poses_local[prev.cylinder_idx] = \
                        self.plan.final_cylinder_poses[prev.cylinder_idx]
                    self.attached_idx = None
        self.current_idx = idx
        self.slider_progress.blockSignals(True)
        self.slider_progress.setValue(idx)
        self.slider_progress.blockSignals(False)
        self.lbl_progress.setText(f"WP: {idx+1} / {len(self.plan.waypoints)}")

    def _on_speed_changed(self, v: int):
        self.lbl_speed.setText(f"{v} fps")
        if self.is_playing:
            self.play_start_time = time.perf_counter()
            self.play_start_idx = self.current_idx

    def on_play(self):
        if self.plan is None or self.is_playing:
            return
        self.is_playing = True
        self._set_config_enabled(False)
        # Si llegó al final, reiniciar
        if self.current_idx >= len(self.plan.waypoints) - 1:
            self._set_current_idx(0)
        self.play_start_time = time.perf_counter()
        self.play_start_idx = self.current_idx
        # Timer de sondeo rápido (~60 Hz) para sincronización con reloj real
        self.anim_timer.setInterval(15)
        self.anim_timer.start()
        self._log("Reproduciendo...", color="#9f9")

    def on_pause(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.anim_timer.stop()
        self._set_config_enabled(True)
        self._log("Pausa.", color="#fa6")

    def on_stop(self):
        self.is_playing = False
        self.anim_timer.stop()
        self._set_config_enabled(True)
        if self.plan is not None:
            self._set_current_idx(0)
            self._render_current_state()
        self._log("Stop.", color="#fa6")

    def on_rewind(self):
        if self.plan is None:
            return
        self._set_current_idx(0)
        self._render_current_state()

    def _anim_step(self):
        if self.plan is None or not self.is_playing:
            self.anim_timer.stop()
            return
        n_wp = len(self.plan.waypoints)
        fps = max(1, self.slider_speed.value())
        elapsed = time.perf_counter() - self.play_start_time
        target_idx = int(self.play_start_idx + elapsed * fps)

        if target_idx >= n_wp - 1:
            # Llegamos al final
            self._set_current_idx(n_wp - 1)
            self._render_current_state()
            self.is_playing = False
            self.anim_timer.stop()
            self._set_config_enabled(True)
            self._log("Trayectoria completa.", color="#9f9")
            return

        if target_idx > self.current_idx:
            self._set_current_idx(target_idx)
            # Log de fallo IK FR5
            if not self.fr5_success_per_wp[self.current_idx]:
                wp = self.plan.waypoints[self.current_idx]
                self._log(f"  WP {self.current_idx+1}: fallo IK FR5 (fase {wp.phase})",
                          color="#fa6")
            self._render_current_state()

    # =========================================================================
    # Modo manual articular de la pinza EndoWrist (shaft, wrist, jaw_dx, jaw_sx)
    # =========================================================================
    def _capture_reference_pose(self):
        """Inicializa la pose articular de referencia."""
        self.q_ref_psm = Q_HOME_LAPARO.copy()
        self.q_manual_psm = Q_HOME_LAPARO.copy()
        if self.q_fr5_init is not None:
            self.q_manual_fr5 = self.q_fr5_init.copy()
            self._sync_fr5_sliders_from_q(self.q_manual_fr5)
        self._reset_manual_sliders_silent()
        self._manual_ik_failed = False

    def _reset_manual_sliders_silent(self):
        """Pone todos los sliders/spins de pinza manual a cero sin disparar callbacks."""
        self._manual_updating = True
        if hasattr(self, 'sliders_endowrist'):
            for key in ("shaft", "wrist", "jaw_dx", "jaw_sx"):
                if key in self.sliders_endowrist:
                    slider, spin = self.sliders_endowrist[key]
                    slider.setValue(0)
                    spin.setValue(0.0)
        self._manual_updating = False

    def on_resetear_pinza(self):
        self._reset_manual_sliders_silent()
        self.q_manual_psm = Q_HOME_LAPARO.copy()
        if self.q_fr5_init is not None:
            self.q_manual_fr5 = self.q_fr5_init.copy()
        self._manual_ik_failed = False
        self._log("Pinza reseteada a 0°.")
        self._render_manual_state()

    def _sync_fr5_sliders_from_q(self, q_fr5: np.ndarray):
        """Sincroniza los sliders y spinboxes del FR5 con una posición articular (rad)."""
        if not hasattr(self, 'sliders_fr5') or len(self.sliders_fr5) < 6:
            return
        self._manual_updating = True
        for i in range(6):
            slider, spin = self.sliders_fr5[i]
            val = float(np.rad2deg(q_fr5[i]))
            spin.setValue(val)
            slider.setValue(int(round(val * slider._scale_factor)))
        self._manual_updating = False

    def on_resetear_fr5(self):
        """Resetea las articulaciones del FR5 a la posición Home."""
        if self.fr5_robot is None:
            return
        self.q_manual_fr5 = self.q_fr5_init.copy()
        self._sync_fr5_sliders_from_q(self.q_manual_fr5)
        self._render_manual_state()

    def _on_fr5_slider_changed(self, idx: int):
        if self._manual_updating or self.fr5_robot is None:
            return
        slider, spin = self.sliders_fr5[idx]
        val = slider.value() / float(slider._scale_factor)
        self._manual_updating = True
        spin.setValue(val)
        self._manual_updating = False
        self._update_fr5_from_manual()

    def _on_fr5_spin_changed(self, idx: int):
        if self._manual_updating or self.fr5_robot is None:
            return
        slider, spin = self.sliders_fr5[idx]
        val = spin.value()
        self._manual_updating = True
        slider.setValue(int(round(val * slider._scale_factor)))
        self._manual_updating = False
        self._update_fr5_from_manual()

    def _update_fr5_from_manual(self):
        """Actualiza q_manual_fr5 leyendo los sliders articulares."""
        if self.fr5_robot is None:
            return
        q_new = np.zeros(6)
        for i in range(6):
            slider, spin = self.sliders_fr5[i]
            q_new[i] = np.deg2rad(spin.value())
        self.q_manual_fr5 = q_new
        self._render_manual_state()


    def _on_endowrist_slider_changed(self, key: str):
        if self._manual_updating:
            return
        slider, spin = self.sliders_endowrist[key]
        val = slider.value() / float(slider._scale_factor)
        self._manual_updating = True
        spin.setValue(val)
        self._manual_updating = False
        self._update_endowrist_from_manual()

    def _on_endowrist_spin_changed(self, key: str):
        if self._manual_updating:
            return
        slider, spin = self.sliders_endowrist[key]
        val = spin.value()
        self._manual_updating = True
        slider.setValue(int(round(val * slider._scale_factor)))
        self._manual_updating = False
        self._update_endowrist_from_manual()

    def _update_endowrist_from_manual(self):
        """Actualiza q_manual_psm con las articulaciones manuales de la pinza EndoWrist."""
        if self.plan is not None:
            return
        shaft_deg = self.sliders_endowrist["shaft"][1].value()
        wrist_deg = self.sliders_endowrist["wrist"][1].value()
        jaw_dx_deg = self.sliders_endowrist["jaw_dx"][1].value()
        jaw_sx_deg = self.sliders_endowrist["jaw_sx"][1].value()

        if self.q_manual_psm is None:
            self.q_manual_psm = Q_HOME_LAPARO.copy()

        # Mover únicamente la pinza: shaft (q4), wrist (q5), y magnitud de apertura (q7)
        self.q_manual_psm[3] = np.deg2rad(shaft_deg)
        self.q_manual_psm[4] = np.deg2rad(wrist_deg)
        self.q_manual_psm[5] = 0.0  # q6 = 0 constante
        self.q_manual_psm[6] = np.deg2rad(max(abs(jaw_dx_deg), abs(jaw_sx_deg)))

        # El FR5 no se mueve en este modo
        if self.q_manual_fr5 is None and self.q_fr5_init is not None:
            self.q_manual_fr5 = self.q_fr5_init.copy()

        self._manual_ik_failed = False
        self._render_manual_state()

    def _render_manual_state(self):
        """Render del PSM y FR5 en el estado manual actual."""
        if self.fr5_robot is None or self.q_manual_psm is None:
            return
        q_psm = self.q_manual_psm
        q_fr5 = self.q_manual_fr5 if self.q_manual_fr5 is not None else self.q_fr5_init
        T_fr5 = fr5_fk(self.fr5_robot, self.fr5_T_tool, q_fr5)

        info_full = (
            f"Modo MANUAL Articular (EndoWrist)\n"
            f"Shaft={self.sliders_endowrist['shaft'][1].value():+.1f}° "
            f"Wrist={self.sliders_endowrist['wrist'][1].value():+.1f}°\n"
            f"JawDX={self.sliders_endowrist['jaw_dx'][1].value():+.1f}° "
            f"JawSX={self.sliders_endowrist['jaw_sx'][1].value():+.1f}°"
        )

        render_full_system(
            self.canvas_full.ax, self.fr5_robot,
            q_fr5, T_fr5, self.spin_dist_trocar.value(), self.p_fulcrum,
            q_psm, self.T_psm_base,
            self.cylinders, self.cyl_poses_local, None,
            ik_failed=self._manual_ik_failed,
            info_text=info_full,
            elev=self.cam_full["elev"], azim=self.cam_full["azim"],
        )
        render_psm_only(
            self.canvas_psm.ax,
            q_psm, self.T_psm_base, self.p_fulcrum,
            self.cylinders, self.cyl_poses_local, None,
            ik_failed=self._manual_ik_failed,
            info_text=info_full,
            elev=self.cam_psm["elev"], azim=self.cam_psm["azim"],
            zoom_factor=self.cam_psm["zoom_factor"],
        )
        self._apply_full_zoom_factor()
        self.canvas_full.draw_idle()
        self.canvas_psm.draw_idle()


# =============================================================================
# Utilidades de rotación
# =============================================================================

def _euler_zyx_to_R(angles_rad: np.ndarray) -> np.ndarray:
    """Convierte ángulos Euler (intrinsic) RX, RY, RZ en matriz de rotación."""
    rx, ry, rz = angles_rad
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _R_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Extrae ángulos Euler RX, RY, RZ (intrinsic) de una matriz de rotación."""
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)
    cy = np.cos(ry)
    if abs(cy) > 1e-6:
        rx = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
        rz = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)
    else:
        # Gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


# =============================================================================
# main
# =============================================================================

def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CoupledSimulatorMainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
