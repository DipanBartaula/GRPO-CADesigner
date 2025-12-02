"""
Script to create simple trimesh-based test scripts
These work without CadQuery dependencies
"""

import os

# Create directory
os.makedirs('test_cad_scripts', exist_ok=True)

# Simple test scripts using only trimesh
simple_scripts = {
    'simple_cube.py': '''"""Test Script: Simple Cube"""
import trimesh
import numpy as np

# Create a simple cube using trimesh primitives
mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
''',

    'sphere.py': '''"""Test Script: Sphere"""
import trimesh

# Create a sphere using icosphere subdivision
mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.5)
''',

    'cylinder.py': '''"""Test Script: Cylinder"""
import trimesh

# Create a cylinder
mesh = trimesh.creation.cylinder(radius=1.0, height=3.0, sections=32)
''',

    'torus.py': '''"""Test Script: Torus"""
import trimesh
import numpy as np

# Create a torus
major_radius = 2.0
minor_radius = 0.5
major_sections = 32
minor_sections = 16

# Generate torus vertices
vertices = []
for i in range(major_sections):
    theta = 2 * np.pi * i / major_sections
    for j in range(minor_sections):
        phi = 2 * np.pi * j / minor_sections
        
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        
        vertices.append([x, y, z])

vertices = np.array(vertices)

# Generate faces
faces = []
for i in range(major_sections):
    for j in range(minor_sections):
        # Current vertex indices
        v1 = i * minor_sections + j
        v2 = i * minor_sections + (j + 1) % minor_sections
        v3 = ((i + 1) % major_sections) * minor_sections + (j + 1) % minor_sections
        v4 = ((i + 1) % major_sections) * minor_sections + j
        
        # Two triangles per quad
        faces.append([v1, v2, v3])
        faces.append([v1, v3, v4])

faces = np.array(faces)

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
''',

    'cone.py': '''"""Test Script: Cone"""
import trimesh

# Create a cone
mesh = trimesh.creation.cone(radius=1.5, height=3.0, sections=32)
''',

    'capsule.py': '''"""Test Script: Capsule"""
import trimesh

# Create a capsule (cylinder with hemispherical ends)
mesh = trimesh.creation.capsule(radius=0.5, height=2.0)
''',

    'complex_composed.py': '''"""Test Script: Complex Composed Shape"""
import trimesh
import numpy as np

# Create a complex shape by combining primitives
# Base cylinder
base = trimesh.creation.cylinder(radius=2.0, height=0.5, sections=32)

# Middle sphere
sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
sphere.apply_translation([0, 0, 1.5])

# Top cone
cone = trimesh.creation.cone(radius=1.0, height=2.0, sections=32)
cone.apply_translation([0, 0, 3.0])

# Combine all meshes
mesh = trimesh.util.concatenate([base, sphere, cone])
''',

    'octahedron.py': '''"""Test Script: Octahedron"""
import trimesh
import numpy as np

# Create an octahedron
vertices = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
]) * 1.5

faces = np.array([
    [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
    [1, 2, 5], [1, 5, 3], [1, 3, 4], [1, 4, 2]
])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
'''
}

# Write all scripts
for filename, content in simple_scripts.items():
    filepath = os.path.join('test_cad_scripts', filename)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")

print(f"\n✓ Created {len(simple_scripts)} test scripts in test_cad_scripts/")