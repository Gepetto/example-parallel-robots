'''
https://github.com/MeMory-of-MOtion/summer-school/blob/master/tutorials/pinocchio/vizutils.py
'''

import meshcat
import numpy as np
import pinocchio as pin

# Meshcat utils

def meshcat_material(r, g, b, a):
    import meshcat

    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))


# Gepetto/meshcat abstraction

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addBox(name, sizex, sizey, sizez, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def addViewerSphere(viz, name, size, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Sphere(size),
                                    meshcat_material(*rgba))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addSphere(name, size, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.applyConfiguration(name, xyzquat)
        viz.viewer.gui.refresh()
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)

def visualizeConstraints(viz, model, data, constraint_models, q=None):
    if q is not None:
        pin.framesForwardKinematics(model, data, q)
        viz.display(q)
    for i, c in enumerate(constraint_models):
        if c.name != '':
            name = c.name
        else:
            name = f"c{i}"
        offset = pin.SE3.Identity()
        offset.translation = np.array([0, 0, 0.005])
        box = addViewerBox(viz, name+"_1", 0.03, 0.02, 0.01, [1, 0, 0, 0.5])
        applyViewerConfiguration(viz, name+"_1", pin.SE3ToXYZQUATtuple(data.oMi[c.joint1_id]*c.joint1_placement.act(offset)))
        box = addViewerBox(viz, name+"_2", 0.03, 0.02, 0.01, [0, 1, 0, 0.5])
        applyViewerConfiguration(viz, name+"_2", pin.SE3ToXYZQUATtuple(data.oMi[c.joint2_id]*c.joint2_placement.act(offset)))