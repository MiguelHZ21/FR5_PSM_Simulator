# Cinemática del PSM da Vinci classic en Python

Implementación en Python de la cinemática directa e inversa del **Patient Side
Manipulator (PSM)** del robot quirúrgico **da Vinci classic (IS-1200)** con la
pinza **EndoWrist**, portada del repositorio de referencia
[`cwru_davinci_kinematics`](https://github.com/cwru-robotics/cwru_davinci_kinematics)
de Case Western Reserve University.

Soporta dos modos de ejecución:

- **Local desde terminal** (script CLI `psm_main.py`).
- **Google Colab / Jupyter** (notebook `psm_test.ipynb`).

## Estructura del proyecto

```
dvrk_psm_kinematics/
├── psm_kinematics.py         # Núcleo: parámetros DH, FK, IK, Jacobiano
├── psm_visualization.py      # Visualización 3D y animación
├── psm_scene.py              # Escena del simulador: mesa, cilindros, agarre
├── psm_pick_place.py         # Planeador pick-and-place pose-a-pose
├── psm_ur5_kinematics.py     # Wrapper UR5 + acople UR5↔PSM
├── psm_main.py               # Script CLI: trayectoria lineal de 2 cm
├── psm_arc_constant.py       # Script CLI: arco con orientación constante
├── psm_arc_rolling.py        # Script CLI: arco con roll variable
├── psm_simulator_gui.py      # Simulador laparoscópico con GUI PyQt6
├── psm_simulator_demo.py     # Demo headless del simulador (genera GIF)
├── psm_ur5_coupled.py        # Sistema acoplado UR5+PSM peg transfer (CLI)
├── psm_ur5_simulator_gui.py  # GUI interactiva del sistema acoplado UR5+PSM
├── psm_ur5_render.py         # Render compartido del sistema acoplado
├── psm_test.ipynb            # Notebook Jupyter / Colab
├── requirements.txt          # Dependencias mínimas
├── documento/                # Reporte técnico LaTeX (32 páginas, 12 figuras)
└── README.md
```

Los módulos `psm_kinematics.py`, `psm_visualization.py`, `psm_scene.py` y
`psm_pick_place.py` son **idénticos** en todos los modos de uso: solo cambian
los scripts/notebook que los importan. El módulo `psm_ur5_kinematics.py`
añade la integración con un brazo industrial UR5 que sostiene el trocar.

---

## Modelo cinemático

- **7 grados de libertad** (3 RCM + muñeca esférica + jaw):
  q = [q1 yaw, q2 pitch, q3 inserción, q4 roll, q5 wrist_pitch, q6 wrist_yaw, q7 jaw]
- Convención **DH modificada (Craig)** con los parámetros del da Vinci classic.
- **Fulcro (RCM)**: el origen del frame base coincide con el Remote Center of
  Motion. La restricción mecánica del paralelogramo del PSM real está
  incorporada implícitamente en la cadena DH.
- **Cinemática inversa**: solver Levenberg-Marquardt amortiguado (DLS) con
  guess analítico inicial y multistart automático.

### Configuración HOME

Existen **dos configuraciones HOME** disponibles:

**HOME vertical** (`Q_HOME`): shaft apuntando recto hacia abajo, configuración pedagógica:

| Articulación | Valor      | Significado                                  |
|--------------|------------|----------------------------------------------|
| q1           | 0 rad      | outer yaw centrado                           |
| q2           | 0 rad      | outer pitch centrado, shaft vertical         |
| q3           | 0.12 m     | inserción para que el TCP esté distal al RCM |
| q4–q6        | 0 rad      | muñeca alineada                              |
| q7           | 0 rad      | jaw cerrado                                  |

TCP en `[0, 0, -0.1234] m`.

**HOME inclinado laparoscópico** (`Q_HOME_TILTED`): shaft inclinado 45° sobre
la horizontal, simulando entrada por trocar real. Solo cambia q2 = π/4:

| Articulación | Valor      | Significado                                  |
|--------------|------------|----------------------------------------------|
| q1           | 0 rad      |                                              |
| q2           | π/4 rad    | outer pitch a 45° → shaft inclinado          |
| q3           | 0.12 m     |                                              |
| q4–q7        | 0 rad      |                                              |

TCP en `[0, -0.0873, -0.0873] m`.

---

## Trayectorias disponibles

### Trayectoria lineal de 2 cm (`psm_main.py`)

Línea de 2 cm a lo largo del eje Z del TCP, partiendo de HOME vertical.
Equivale a avanzar la herramienta a través del trocar.

### Arco semicircular en plano XY del TCP — Caso 1 (`psm_arc_constant.py`)

Arco semicircular de radio 1 cm centrado en HOME inclinado, contenido en el
plano XY del TCP (perpendicular al shaft inclinado a 45°). La pinza
**mantiene su orientación constante** durante todo el recorrido. Todas las
articulaciones se coordinan para que el TCP describa el arco mientras la
pinza apunta siempre en la misma dirección.

### Arco semicircular en plano XY del TCP — Caso 2 (`psm_arc_rolling.py`)

Idéntico al caso 1, pero la pinza **rota linealmente 40° alrededor de su
eje Z** durante el recorrido (parámetro `--roll-deg`). Representa el caso
clínico de iniciar un movimiento de sutura curva.

### Simulador laparoscópico tipo "peg transfer" (`psm_simulator_gui.py`)

Simulador interactivo con **interfaz gráfica de usuario PyQt6** que reproduce
la tarea clásica de transferencia de objetos del entrenamiento laparoscópico
(FLS Peg Transfer):

1. Genera 3 cilindros aleatorios sobre una mesa de trabajo, con orientación
   vertical o acostada al azar.
2. Calcula trayectorias pick-and-place para cada cilindro, resolviendo IK
   para las 8 fases canónicas: `approach → grasp → close → lift → transit
   → lower → release → retract`.
3. Coloca cada cilindro **vertical 2 cm más adelante** (configurable) de su
   posición original.
4. Anima la ejecución completa con efecto de agarre real (mordazas que se
   abren y cierran, cilindros que viajan pegados al TCP).

Si la escena aleatoria no es alcanzable por completo, el sistema regenera
automáticamente hasta encontrar una configuración válida (máximo 20
reintentos).

### Demo headless del simulador (`psm_simulator_demo.py`)

Versión sin GUI del simulador: genera el plan completo y produce un GIF
animado, útil para servidores sin display, presentaciones y reportes.

### Sistema acoplado UR5 + PSM (`psm_ur5_coupled.py`)

Acopla un brazo industrial Universal Robots UR5 que sostiene un trocar
laparoscópico, con el PSM EndoWrist atravesando el trocar para ejecutar
la tarea de peg transfer. Reproduce de forma fiel el escenario clínico
en el que un brazo robótico colaborativo posiciona el portatrocar y el
EndoWrist trabaja a través de él, con el fulcro físicamente clavado en
la pared del paciente.

Funcionamiento:

1. **Inicialización**: el usuario elige una de dos formas de definir
   dónde queda el fulcro:
   - `--mode home`: usa el HOME del UR5 y el fulcro emerge a partir de
     la pose de su efector y la longitud del trocar.
   - `--mode fulcrum --fulcrum X Y Z --shaft-axis NX NY NZ`: el usuario
     especifica directamente la posición y dirección del fulcro deseados,
     y el UR5 resuelve IK para llegar ahí.
2. **Planeación PSM**: se genera la escena aleatoria de cilindros y se
   planean las 8 fases pick-and-place exactamente como en el simulador
   estándar.
3. **Proyección PSM → UR5**: para cada uno de los ~240 waypoints del PSM,
   se calcula la pose que debe tener el TCP del UR5 para que el trocar
   esté coherentemente alineado con el shaft en esa configuración del
   PSM. El fulcro permanece fijo en el mundo (verificado a precisión
   numérica de 10⁻¹³ mm).
4. **Salidas**: dos GIFs (sistema completo con UR5+trocar+PSM+cilindros,
   y vista zoom solo del PSM), gráfica de errores IK del UR5, y dos
   archivos TXT con las trayectorias articulares (UR5 y PSM por separado)
   listos para ejecución en hardware real vía RTDE.

```bash
# Modo HOME del UR5 (default)
python psm_ur5_coupled.py

# Modo fulcro específico
python psm_ur5_coupled.py --mode fulcrum --fulcrum 0.55 -0.10 0.45 \
                          --shaft-axis 0.5 0.0 -0.866

# Cambiar distancia del trocar
python psm_ur5_coupled.py --dist-trocar 0.20

# Sin GIFs (solo plan + errores + TXT, mucho más rápido)
python psm_ur5_coupled.py --no-gif
```

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--mode {home,fulcrum}` | Modo de inicialización | `home` |
| `--fulcrum X Y Z` | Posición del fulcro [m] (modo fulcrum) | `0.55 -0.10 0.40` |
| `--shaft-axis NX NY NZ` | Dirección del shaft (se normaliza) | `0.5 0 -0.866` |
| `--dist-trocar` | Distancia TCP UR5 → fulcro [m], rango 0.05–0.50 | `0.15` |
| `--n-cyl` | Número de cilindros | `3` |
| `--seed` | Seed aleatoria de la escena | aleatoria |
| `--offset-x` | Desplazamiento X de colocación [m] | `0.020` |
| `--fps` | FPS de los GIFs | `12` |
| `--step` | Submuestreo de waypoints en GIFs | `3` |
| `--no-gif` | No generar GIFs | desactivado |
| `--output-dir` | Directorio de salida | `./resultados_ur5_psm` |

**Salidas en el directorio**:

| Archivo | Contenido |
|---------|-----------|
| `sistema_completo.gif` | UR5 + trocar + PSM + cilindros + mesa, vista global |
| `psm_solo.gif` | Vista zoom alrededor del fulcro, sin UR5 |
| `errores_ur5.png` | Error IK UR5 vs waypoint |
| `posiciones_articulares_ur5.txt` | q1..q6 del UR5 por waypoint, anotado por fase |
| `posiciones_articulares_psm.txt` | q1..q7 del PSM por waypoint, anotado por fase |

**Dependencias adicionales**:

```bash
pip install roboticstoolbox-python spatialmath-python
# Para enviar a robot real (opcional):
pip install ur-rtde
```

Los TXT exportados sirven directamente para ejecución en hardware: cada
fila representa un waypoint sincronizado entre UR5 y PSM, con su fase y
estado de agarre, listo para enviar vía RTDE al UR5 y vía dVRK al PSM.

### GUI interactiva del sistema acoplado (`psm_ur5_simulator_gui.py`)

Versión gráfica interactiva del sistema acoplado UR5+PSM, basada en PyQt6.
Una sola ventana con dos paneles 3D lado a lado: a la izquierda el sistema
completo (UR5 + trocar + PSM + cilindros + mesa) y a la derecha la vista
zoom solo del PSM y la zona de trabajo. Los paneles se sincronizan
automáticamente con el reloj de animación.

```bash
python psm_ur5_simulator_gui.py
```

**Controles disponibles**:

- **Configuración de la escena**: número de cilindros (1–5), seed con
  botón "aleatoria", offsets X/Y de colocación.
- **Configuración UR5 + trocar**: modo de inicialización (HOME UR5 o
  fulcro especificado), coordenadas/dirección del fulcro (en modo
  fulcrum), distancia trocar (TCP UR5→fulcro, rango 5–50 cm), longitud
  total de la pinza.
- **Acciones**: `[Reset]`, `[Generar escena]`, `[Calcular trayectorias]`,
  `[Exportar TXT]` (genera tres archivos con timestamp en la carpeta
  elegida: articulares UR5, articulares PSM, cartesianas con drift del
  fulcro).
- **Reproducción**: `[⏮]` rewind, `[⏹]` stop, `[⏸]` pause, `[▶]` play,
  slider de velocidad (1–60 fps), slider de progreso arrastrable para
  saltar a cualquier waypoint.
- **Cámara y zoom por panel**: sliders de elevación y azimut, botones
  `[+]` y `[-]` para zoom, rueda del ratón para zoom natural.

**Comportamiento**:

- La configuración se bloquea automáticamente durante la animación;
  hay que pulsar `[Stop]` para modificar cualquier parámetro.
- Cambiar el modo de inicialización limpia la escena y plan, volviendo
  al estado inicial.
- Los fallos de IK del UR5 (cuando ocurren) se anotan en el log con un
  indicador visual (TCP en rojo) pero la animación continúa sin
  interrupción.
- La ventana es completamente adaptable; el splitter horizontal permite
  redimensionar paneles 3D vs panel de controles.

Ideal para enseñanza, exploración paramétrica (ver el efecto de cambiar
`dist_trocar`, la posición del fulcro, etc.) y validación visual antes
de exportar trayectorias para hardware real.

---

## Uso local desde terminal

### 1. Instalación

#### Opción A — usando entorno virtual de Python (recomendado)

En Linux / macOS:

```bash
# Entrar al directorio del proyecto
cd dvrk_psm_kinematics

# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

En Windows (cmd o PowerShell):

```powershell
cd dvrk_psm_kinematics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Opción B — usando conda

```bash
cd dvrk_psm_kinematics
conda create -n dvrk python=3.11 numpy matplotlib pillow -y
conda activate dvrk
```

#### Opción C — instalación directa sin entorno virtual

Si tienes Python 3.10+ y no quieres aislar el entorno:

```bash
pip install numpy matplotlib pillow
```

### 2. Ejecución desde terminal

#### Trayectoria lineal de 2 cm

```bash
python psm_main.py                    # ejecución básica con ventanas
python psm_main.py --no-show          # solo guardar archivos
python psm_main.py --no-show --save-gif
```

#### Caso 1: Arco con orientación constante

```bash
python psm_arc_constant.py
python psm_arc_constant.py --no-show --save-gif
python psm_arc_constant.py --radius 0.015 --arc-deg 180 --points 60 --output-dir ./caso1
python psm_arc_constant.py --help
```

Opciones específicas:
| Opción | Descripción | Valor por defecto |
|--------|-------------|-------------------|
| `--radius` | Radio del arco [m] | 0.01 |
| `--arc-deg` | Ángulo total del arco [grados] | 180.0 |
| `--points` | Número de waypoints | 60 |
| `--n-triads` | Triedros XYZ a superponer en el gráfico | 10 |
| `--output-dir` | Directorio de salida | `./resultados_arco_caso1` |

#### Caso 2: Arco con rotación de la pinza (roll)

```bash
python psm_arc_rolling.py
python psm_arc_rolling.py --no-show --save-gif
python psm_arc_rolling.py --roll-deg 60 --radius 0.012
python psm_arc_rolling.py --help
```

Opciones específicas: las mismas del caso 1, más:
| Opción | Descripción | Valor por defecto |
|--------|-------------|-------------------|
| `--roll-deg` | Rotación total del roll [grados] | 40.0 |

#### Simulador laparoscópico con GUI (`psm_simulator_gui.py`)

Requiere PyQt6 instalado (incluido en `requirements.txt`).

```bash
python psm_simulator_gui.py
```

Abre una ventana con:
- **Vista 3D** de la escena (robot, mesa, cilindros).
- **Botón [1. Generar cilindros]** — crea 3 cilindros aleatorios.
- **Botón [2. Calcular trayectorias]** — planifica IK pick-and-place.
- **Botón [3. Ejecutar]** — anima la ejecución completa.
- **Botón [Reset]** — vuelve a HOME y limpia escena.
- **Slider** de velocidad de animación (1–50 fps).
- **Log** con los eventos de cada fase.
- **Spinboxes** para configurar número de cilindros y desplazamiento de colocación (X, Y).

Flujo típico de uso:

1. Pulsar `1. Generar cilindros` — aparecen 3 cilindros aleatorios sobre la mesa.
2. Pulsar `2. Calcular trayectorias` — el sistema planifica todo el pick-and-place.
3. Pulsar `3. Ejecutar` — el robot se mueve y transfiere los cilindros uno por uno.

Si la escena aleatoria no es alcanzable, el botón 2 regenera automáticamente
hasta encontrar una válida.

#### Simulador headless (`psm_simulator_demo.py`)

Versión sin GUI que genera un GIF animado:

```bash
python psm_simulator_demo.py
python psm_simulator_demo.py --seed 42 --output demo.gif --fps 15
python psm_simulator_demo.py --n-cyl 3 --offset-x 0.025 --offset-y 0.005
python psm_simulator_demo.py --no-gif    # solo PNG estáticos
```

Opciones:
| Opción | Descripción | Valor por defecto |
|--------|-------------|-------------------|
| `--n-cyl` | Número de cilindros | 3 |
| `--offset-x` | Desplazamiento X al colocar [m] | 0.020 |
| `--offset-y` | Desplazamiento Y al colocar [m] | 0.000 |
| `--seed` | Seed aleatoria (None = aleatoria) | None |
| `--output` | Archivo GIF de salida | `psm_simulador_demo.gif` |
| `--fps` | FPS de la animación | 12 |
| `--step` | Submuestreo de waypoints | 2 |
| `--no-gif` | Solo PNG estáticos, sin GIF | desactivado |
| `--max-retries` | Reintentos al regenerar escena | 20 |

### 3. Archivos generados

#### Trayectoria lineal (`psm_main.py` → `./resultados_psm/`)

| Archivo | Contenido |
|---------|-----------|
| `01_pose_home.png` | PSM en HOME vertical con trayectoria objetivo |
| `02_comparacion_inicio_final.png` | Vista lado a lado: pose inicial vs pose final |
| `03_trayectorias_articulares.png` | Curvas q1..q7 a lo largo de la línea |
| `04_errores_cartesianos.png` | Error por componente y norma del error cartesiano |
| `05_animacion.gif` | Animación 3D (solo con `--save-gif`) |

#### Arco caso 1 / caso 2 (→ `./resultados_arco_casoN/`)

| Archivo | Contenido |
|---------|-----------|
| `01_pose_home_inclinada.png` | PSM en HOME inclinado 45° con arco objetivo |
| `02_deseada_vs_ejecutada.png` | Vista global: trayectoria deseada vs ejecutada |
| `02b_deseada_vs_ejecutada_zoom.png` | Zoom al arco con triedros XYZ |
| `02c_vista_frontal[_roll].png` | Vista perpendicular al plano del arco |
| `03_inicio_fin.png` | Comparación lado a lado de las poses inicial y final |
| `04_trayectorias_articulares.png` | Curvas q1..q7 a lo largo del arco |
| `05_errores_cartesianos.png` | Error cartesiano por waypoint |
| `06_animacion_casoN.gif` | Animación 3D (solo con `--save-gif`) |

---

## Uso en Google Colab

1. En [colab.research.google.com](https://colab.research.google.com) crear un
   notebook nuevo o subir `psm_test.ipynb`.
2. En el panel lateral *Files*, subir también `psm_kinematics.py` y
   `psm_visualization.py` al directorio raíz de la sesión.
3. Ejecutar todas las celdas en orden (`Runtime > Run all`).

Las dependencias (numpy, matplotlib) ya están preinstaladas en Colab.

---

## Uso en Jupyter local

Si prefieres usar el notebook localmente en lugar del script de terminal:

```bash
pip install jupyter
jupyter notebook psm_test.ipynb
```

o con JupyterLab:

```bash
pip install jupyterlab
jupyter lab psm_test.ipynb
```

---

## Uso programático (importación como librería)

Los módulos pueden usarse desde cualquier código Python:

```python
from psm_kinematics import (
    Q_HOME, forward_kinematics, inverse_kinematics,
    linear_trajectory_along_tcp_axis, solve_trajectory_ik,
)

# 1. FK en HOME
T_home = forward_kinematics(Q_HOME)
print(T_home)

# 2. Trayectoria de 2 cm en Z del TCP
s, poses = linear_trajectory_along_tcp_axis(
    Q_HOME, length=0.02, n_points=50, axis="z"
)

# 3. Resolver IK con warm-start
Q_traj, results = solve_trajectory_ik(poses, Q_HOME)

# 4. IK individual
import numpy as np
T_target = np.eye(4)  # alguna pose homogénea 4x4
T_target[:3, 3] = [0.0, 0.0, -0.13]
res = inverse_kinematics(T_target, q_init=Q_HOME)
print(f"Convergencia: {res.success}, error: {res.pos_error:.2e} m")
print(f"Solución articular: {res.q}")
```

---

## Resultados de validación

Sobre la trayectoria de 2 cm con 50 waypoints:

- IK exitosa en **49/49** puntos.
- Error cartesiano máximo: **~4 × 10⁻¹⁰ m** (sub-nanómetro, precisión de máquina).
- Iteraciones por waypoint: **2** (gracias al warm-start).
- Variación articular: **únicamente q3 cambia (+2 cm)**, lo cual es físicamente
  correcto — avanzar a lo largo del eje Z del TCP en HOME equivale a empujar
  la herramienta a través del trocar.

Sobre 30 round-trips aleatorios cubriendo todo el espacio articular:

- Tasa de éxito: **30/30**.
- Error de posición máximo: **~7 × 10⁻⁷ m** (sub-micrómetro).
- Error rotacional máximo: **~3 × 10⁻⁶ rad**.

---

## Solución de problemas

**`ModuleNotFoundError: No module named 'psm_kinematics'`**

El script `psm_main.py` debe ejecutarse desde el directorio que contiene los
módulos `.py`. Verifica con `ls` (Linux/Mac) o `dir` (Windows) que estás en el
directorio correcto.

**Las ventanas de matplotlib no se abren en SSH / WSL / servidor remoto**

Usa `--no-show` para evitar abrir ventanas y solo guardar las figuras:

```bash
python psm_main.py --no-show --save-gif
```

**El GIF tarda mucho en guardarse**

Reduce `--points` (menos waypoints) o `--gif-fps`:

```bash
python psm_main.py --no-show --save-gif --points 25 --gif-fps 8
```

**Error con `pillow` al guardar GIF**

```bash
pip install --upgrade pillow
```

**En Linux: backend `tkinter` no disponible**

Si aparece un error sobre `tkinter` al intentar mostrar ventanas:

```bash
sudo apt install python3-tk      # Debian / Ubuntu
sudo dnf install python3-tkinter # Fedora
```

O simplemente usa `--no-show` para evitar el problema.

---

## Licencia

Trabajo derivado del repositorio CWRU original (GPL-3.0). Autor del port:
Hermes Fabián Vargas Rosero, Universidad del Cauca.
