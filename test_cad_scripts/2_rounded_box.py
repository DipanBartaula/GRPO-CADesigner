"""
CADQuery Test Script 2: Rounded Box
Creates a box with rounded edges using fillets
"""

import cadquery as cq
import trimesh
import numpy as np

try:
    # Create a box with rounded edges
    result = (
        cq.Workplane("XY")
        .box(4, 3, 2)
        .edges("|Z")
        .fillet(0.3)
    )
    
    # Export to STL and load as trimesh
    import tempfile
    import os
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Export to STL file
        cq.exporters.export(result, tmp_path)
        
        # Load with trimesh
        mesh = trimesh.load(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"Export failed: {e}")
        # Fallback to simple box
        mesh = trimesh.creation.box(extents=[4, 3, 2])
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

except Exception as e:
    print(f"CadQuery creation failed: {e}")
    # Fallback mesh
    mesh = trimesh.creation.box(extents=[4, 3, 2])