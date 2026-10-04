# Especificación Técnica y Guía de Operación: Pinza EndoWrist

Este documento proporciona una especificación completa del instrumento quirúrgico simulado **EndoWrist**, sus articulaciones, rangos de movimiento, convenciones de unidades, comunicación ROS 2 y formatos de archivos para la generación automática de trayectorias.

---

## 1. Visión General de la Pinza EndoWrist

La pinza **EndoWrist** es un mecanismo quirúrgico de **4 Grados de Libertad (DoF)** montado sobre el efector final del robot articular FR5. Permite realizar maniobras de orientación fina y agarre/apertura dentro del campo quirúrgico.

```
       [Efector Final FR5]
               │
           ( shaft )   <- Rota la herramienta (Tool Roll) en X local
               │
           ( wrist )   <- Flexiona la muñeca (Wrist Pitch) en Z local
            ┌──┴──┐
       (jaw_dx) (jaw_sx) <- Mandíbulas derecha/izquierda en Z local
```

---

## 2. Articulaciones y Rangos de Movimiento

La pinza consta de 4 articulaciones controladas independientemente:

| Nombre ROS | Nombre Unity | Eje de Rotación Local | Rango en Grados (°) | Rango en Radianes (rad) | Descripción / Función |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `shaft` | `Endo_eje` | **X** | $[-180^\circ, +180^\circ]$ | $[-\pi, +\pi]$ | Rotación axial del vástago (Tool Roll) |
| `wrist` | `Endo_muneca` | **Z** | $[-70^\circ, +70^\circ]$ | $[-1.2217, +1.2217]$ | Inclinación/flexión de la muñeca (Pitch) |
| `jaw_dx` | `Endo_mandibula_dx` | **Z** | $[-60^\circ, +60^\circ]$ | $[-1.0472, +1.0472]$ | Apertura/movimiento mandíbula derecha |
| `jaw_sx` | `Endo_mandibula_sx` | **Z** | $[-60^\circ, +60^\circ]$ | $[-1.0472, +1.0472]$ | Apertura/movimiento mandíbula izquierda |

---

## 3. Convenciones de Unidades y Transformación

Para garantizar la interoperabilidad entre capas del sistema:

1. **Unity, UI y Archivos TXT (`ArticularPoints.txt`, `CartesianPoints.txt`)**:
   * **Unidad**: **GRADOS ($^\circ$)**.
   * Todos los campos de texto, sliders y puntos guardados se leen y escriben en grados.

2. **Capa ROS 2 (`sensor_msgs/msg/JointState`, Simulador, Emulador)**:
   * **Unidad**: **RADIANES ($\text{rad}$)**.

3. **Ecuaciones de Conversión**:
   $$\text{Ángulo}_{\text{rad}} = \text{Ángulo}_{\text{deg}} \times \frac{\pi}{180}$$
   $$\text{Ángulo}_{\text{deg}} = \text{Ángulo}_{\text{rad}} \times \frac{180}{\pi}$$

---

## 4. Estructura de Archivos de Trayectoria (`TXT`)

Si deseas generar archivos de puntos compatibles con el sistema (`fr5_ik_node.py` o Unity), deben seguir las siguientes estructuras exactas.

### 4.1. Archivo Articular (`ArticularPoints.txt`)
```text
articular
X_mgd,Y_mgd,Z_mgd,Rx_mgd,Ry_mgd,Rz_mgd,speed,delay,shaft_deg,wrist_deg,jaw_dx_deg,jaw_sx_deg
```

* **Línea 1**: La palabra estricta `articular`.
* **Líneas subsecuentes**: Valores separados por comas (CSV):
  * `Col 0-5`: Posición cartesiana calculada por Cinemática Directa (MGD) en mm y grados (`X, Y, Z, Rx, Ry, Rz`).
  * `Col 6`: Velocidad de ejecución (ej. `10.0`).
  * `Col 7`: Delay/retardo en segundos (ej. `0.0`).
  * `Col 8`: `shaft` en grados (ej. `0.000000`).
  * `Col 9`: `wrist` en grados (ej. `30.000000`).
  * `Col 10`: `jaw_dx` en grados (ej. `15.000000`).
  * `Col 11`: `jaw_sx` en grados (ej. `15.000000`).

### 4.2. Archivo Cartesiano (`CartesianPoints.txt`)
```text
cartesiano
X,Y,Z,Rx,Ry,Rz,speed,delay,shaft_deg,wrist_deg,jaw_dx_deg,jaw_sx_deg
```

---

## 5. Protocolos de Comunicación ROS 2

Para controlar o escuchar la pinza desde un script externo en ROS 2:

### 5.1. Tópico `/endowrist` (Publicador/Suscriptor de Estado)
* **Tipo de Mensaje**: `sensor_msgs/msg/JointState`
* **Estructura del Mensaje**:
  ```python
  msg = JointState()
  msg.name = ["shaft", "wrist", "jaw_dx", "jaw_sx"]
  msg.position = [shaft_rad, wrist_rad, jaw_dx_rad, jaw_sx_rad]
  ```

### 5.2. Tópico `/input_endo_wrist` (Comandos directos desde Unity)
* **Tipo de Mensaje**: `std_msgs/msg/String`
* **Formato**: `"shaft_deg,wrist_deg,jaw_dx_deg,jaw_sx_deg"` (Valores en grados en texto separado por comas).

---

## 6. Ejemplo en Python para Generar Trayectorias Automáticas

A continuación se presenta un script plantilla en Python que puedes adaptar para crear trayectorias automáticas con la pinza EndoWrist:

```python
import math

def generate_endowrist_trajectory(output_filename="ArticularPoints.txt", num_points=50):
    """
    Genera un archivo de trayectoria articular simulado con movimiento suave de la pinza.
    """
    with open(output_filename, "w") as f:
        # Encabezado
        f.write("articular\n")
        
        for i in range(num_points):
            t = i / float(num_points - 1)
            
            # Poses ficticias del robot FR5 (MGD)
            x, y, z = -308.63, -296.19, 145.81
            rx, ry, rz = -160.03, 6.49, -177.31
            speed = 10.0
            delay = 0.0
            
            # Generación de perfil de movimiento para EndoWrist (en GRADOS)
            shaft_deg = 45.0 * math.sin(2 * math.pi * t)        # Oscila entre -45° y +45°
            wrist_deg = 30.0 * math.cos(math.pi * t)            # Transición entre -70° y +70°
            jaw_dx_deg = 20.0 + 15.0 * math.sin(4 * math.pi * t)# Rango [-60°, +60°]
            jaw_sx_deg = jaw_dx_deg                              # Movimiento simétrico
            
            # Escribir línea CSV
            line = (
                f"{x:.2f},{y:.2f},{z:.2f},{rx:.2f},{ry:.2f},{rz:.2f},"
                f"{speed:.1f},{delay:.1f},"
                f"{shaft_deg:.6f},{wrist_deg:.6f},{jaw_dx_deg:.6f},{jaw_sx_deg:.6f}\n"
            )
            f.write(line)
            
    print(f"Trayectoria generada exitosamente en '{output_filename}' con {num_points} puntos.")

if __name__ == "__main__":
    generate_endowrist_trajectory()
```

---

## 7. Resumen de Reglas de Oro

1. **Grados en Archivos/UI**: Al escribir o modificar archivos `.txt` o interactuar con Unity, utiliza **siempre GRADOS**.
2. **Radianes en ROS**: Al publicar en tópicos ROS 2 (`sensor_msgs/JointState`), convierte previamente a **RADIANES**.
3. **Respetar Límites**:
   * `shaft`: $[-180^\circ, 180^\circ]$ ($\approx [-3.1416, 3.1416]\text{ rad}$)
   * `wrist`: $[-70^\circ, 70^\circ]$ ($\approx [-1.2217, 1.2217]\text{ rad}$)
   * `jaw_dx` / `jaw_sx`: $[-60^\circ, 60^\circ]$ ($\approx [-1.0472, 1.0472]\text{ rad}$)
