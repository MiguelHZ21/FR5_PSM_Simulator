#!/usr/bin/env python3
"""
verificar_instalacion.py
=========================

Comprueba que todos los módulos del proyecto y las dependencias externas
están correctamente instalados. Ejecutar antes de la GUI principal:

    python verificar_instalacion.py

Si todo está bien, imprime "TODO OK" al final. Si algo falla, indica
exactamente qué falta y cómo arreglarlo.
"""

import sys
import importlib


def check(label, fn):
    """Ejecuta una verificación y muestra OK/FALLO."""
    try:
        result = fn()
        msg = f"  [OK]   {label}"
        if result:
            msg += f"  ({result})"
        print(msg)
        return True
    except Exception as e:
        print(f"  [FALLA] {label}")
        print(f"          -> {type(e).__name__}: {e}")
        return False


def main() -> int:
    print()
    print("=" * 70)
    print(" Verificación del sistema completo PSM + UR5")
    print("=" * 70)

    all_ok = True

    print("\n[1/3] Versión de Python")
    py_version = sys.version_info
    if py_version < (3, 10):
        print(f"  [FALLA] Python {py_version.major}.{py_version.minor} "
              f"(se requiere 3.10+)")
        all_ok = False
    else:
        print(f"  [OK]   Python {py_version.major}.{py_version.minor}."
              f"{py_version.micro}")

    print("\n[2/3] Dependencias externas")
    deps = [
        ("numpy", "numpy", lambda m: m.__version__),
        ("matplotlib", "matplotlib", lambda m: m.__version__),
        ("PIL (Pillow)", "PIL", lambda m: getattr(m, "__version__", "?")),
        ("PyQt6", "PyQt6.QtCore", lambda m: m.PYQT_VERSION_STR),
        ("roboticstoolbox-python", "roboticstoolbox",
         lambda m: getattr(m, "__version__", "?")),
        ("spatialmath-python", "spatialmath",
         lambda m: getattr(m, "__version__", "?")),
    ]
    for label, modname, getver in deps:
        ok = check(label, lambda: getver(importlib.import_module(modname)))
        if not ok:
            all_ok = False

    # Verificación específica de NumPy < 2
    try:
        import numpy as np
        major = int(np.__version__.split(".")[0])
        if major >= 2:
            print(f"  [AVISO] NumPy {np.__version__} es 2.x; "
                  f"roboticstoolbox no es compatible.")
            print(f"          Solución:  pip install 'numpy<2' --force-reinstall")
            all_ok = False
    except Exception:
        pass

    print("\n[3/3] Módulos del proyecto (en esta carpeta)")
    modules = [
        "psm_kinematics",
        "psm_scene",
        "psm_pick_place",
        "psm_visualization",
        "psm_ur5_kinematics",
        "psm_ur5_render",
    ]
    for modname in modules:
        ok = check(modname, lambda m=modname: importlib.import_module(m).__name__)
        if not ok:
            all_ok = False

    print()
    print("=" * 70)
    if all_ok:
        print(" TODO OK — puedes ejecutar:  python psm_ur5_simulator_gui.py")
    else:
        print(" Hay errores. Revisa los mensajes [FALLA] arriba.")
        print(" Consulta README_simple.md sección 'Resolución de problemas'.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
