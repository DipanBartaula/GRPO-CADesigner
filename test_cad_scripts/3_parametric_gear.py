"""
CADQuery Test Script 3: Parametric Gear
Creates a simple gear with customizable parameters
"""

import cadquery as cq
import trimesh
import numpy as np
import math

try:
    # Gear parameters
    num_teeth = 12
    tooth_height = 0.3
    inner_radius = 1.0
    outer_radius = 1.5
    thickness = 0.5
    
    # Create gear profile
    points = []
    for i in range(num_teeth):
        # Angle for this tooth
        angle1 = 2 * math.pi * i / num_teeth
        angle2 = 2 * math.pi * (i + 0.3) / num_teeth
        angle3 = 2 * math.pi * (i + 0.5) / num_teeth
        angle4 = 2 * math.pi * (i + 0.8) / num_teeth
        
        # Inner arc
        points.append((inner_radius * math.cos(angle1), inner_radius * math.sin(angle1)))
        
        # Tooth rise
        points.append((outer_radius * math.cos(angle2), outer_radius * math.sin(angle2)))
        
        # Tooth top
        points.append((outer_radius * math.cos(angle3), outer_radius * math.sin(angle3)))
        
        # Tooth fall
        points.append((inner_radius * math.cos(angle4), inner_radius * math.sin(angle4)))
    
    # Create gear using CadQuery
    result = (
        cq.Workplane("XY")
        .polyline(points)
        .close()
        .extrude(thickness)
        .faces(">Z")
        .workplane()
        .hole(0.5)
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
        # Fallback to cylinder
        mesh = trimesh.creation.cylinder(radius=outer_radius, height=thickness, sections=num_teeth*4)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

except Exception as e:
    print(f"Gear creation failed: {e}")
    # Fallback mesh
    mesh = trimesh.creation.cylinder(radius=1.5, height=0.5, sections=24)