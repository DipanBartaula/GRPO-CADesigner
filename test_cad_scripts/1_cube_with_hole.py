"""
CADQuery Test Script 1: Cube with Cylindrical Hole
Creates a cube with a hole drilled through it
"""

import cadquery as cq
import trimesh

# Create a cube with a hole
result = (
    cq.Workplane("XY")
    .box(3, 3, 3)
    .faces(">Z")
    .workplane()
    .hole(1.0)
)

# Export to trimesh
# Get the shape and convert to mesh
shape = result.val()
vertices = []
faces = []

try:
    # Try to export via STL string
    import io
    from stl import mesh as stl_mesh
    
    stl_str = result.exportStl()
    
    # Parse STL and create trimesh
    stl_file = io.BytesIO(stl_str.encode() if isinstance(stl_str, str) else stl_str)
    mesh = trimesh.load(stl_file, file_type='stl')
    
except Exception as e:
    # Fallback: Create a simple box mesh
    print(f"CadQuery export failed, using fallback: {e}")
    mesh = trimesh.creation.box(extents=[3, 3, 3])