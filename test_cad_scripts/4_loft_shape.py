"""
CADQuery Test Script 4: Lofted Shape
Creates a complex shape by lofting between different profiles
"""

import cadquery as cq
import trimesh
import numpy as np

try:
    # Create a lofted shape - bottle-like object
    result = (
        cq.Workplane("XY")
        # Base circle
        .circle(2.0)
        .workplane(offset=2.0)
        # Middle square
        .rect(2.5, 2.5)
        .workplane(offset=2.0)
        # Top circle (smaller)
        .circle(1.0)
        # Loft between the profiles
        .loft(combine=True)
    )
    
    # Export to trimesh
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cq.exporters.export(result, tmp_path)
        mesh = trimesh.load(tmp_path)
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"Export failed: {e}")
        # Fallback to cone
        mesh = trimesh.creation.cone(radius=2.0, height=4.0, sections=32)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

except Exception as e:
    print(f"Loft creation failed: {e}")
    # Fallback mesh
    mesh = trimesh.creation.cone(radius=2.0, height=4.0, sections=32)