"""
-*- coding: utf-8 -*-
Virgile BATTO & Ludovic DE MATTEIS - April 2023

Tools to merge and split configuration into actuated and non-actuated parts. Also contains tools to freeze joints from a model
"""

import numpy as np
import pinocchio as pin

def freezeJoints(model, constraint_models, actuation_model, visual_model, collision_model, indexToLock, reference=None):
    '''
    Reduce the model by freezing all joint needed.
    Argument:
        model - Pinocchio robot model
        constraint_models - Pinocchio robot constraint models list
        actuation_model - robot actuation model
        visual_model - Pinocchio robot visual model
        collision_model - Pinocchio robot collision model
        indexToLock: indexes of the joints to lock
        reference - reference configuration to reduce the model from, fixed joints will get their reference configuration fixed
    Return:
        reduced_model - Reduced Pinocchio robot model
        reduced_constraint_models - Reduced Pinocchio robot constraint models list
        reduced_actuation_model - Reduced robot actuation model
        reduced_visual_model - Reduced Pinocchio robot visual model
        reduced_collision_model - Reduced Pinocchio robot collision model
    '''
    if reference is None:
        reference = pin.neutral(model)
    print('Reducing the model')
    reduced_model, (reduced_visual_model, reduced_collision_model) = \
        pin.buildReducedModel(
            model, [visual_model, collision_model], indexToLock, reference)

    if constraint_models is not None:
        print('Reducing the constraint models')
        toremove = []
        for cm in constraint_models:
            print(cm.name)

            # In this function we make the huge assumption that if the parent joint of a constrained frame is fixed is grand-parent is not
            n1 = model.names[cm.joint1_id]
            n2 = model.names[cm.joint2_id]
            idf1 = reduced_model.getFrameId(n1)
            idf2 = reduced_model.getFrameId(n2)
            print(idf1, idf2)

            # Looking for corresponding new frame (ie is the joint fixed ?)
            if idf1 < reduced_model.nframes:
                # The frame has been found
                f1 = reduced_model.frames[idf1]
                cm.joint1_id = f1.parentJoint
                cm.joint1_placement = f1.placement*cm.joint1_placement # Update placement
            else:
                # The joint has not been freezed
                idj1 = reduced_model.getJointId(n1)
                cm.joint1_id = idj1
            # Same for j2
            if idf2 < reduced_model.nframes:
                # The frame has been found
                f2 = reduced_model.frames[idf2]
                cm.joint2_id = f2.parentJoint
                cm.joint2_placement = f2.placement*cm.joint2_placement # Update placement
            else:
                # The joint has not been freezed
                idj2 = reduced_model.getJointId(n2)
                cm.joint2_id = idj2

            if cm.joint1_id == cm.joint2_id:
                toremove.append(cm)
                print(f'Remove constraint {n1}//{n2} (during freeze)')
        reduced_constraint_models = [
            pin.RigidConstraintModel(
                cm.type,
                model,
                cm.joint1_id,
                cm.joint1_placement,
                cm.joint2_id,  # To the world
                cm.joint2_placement,
                pin.ReferenceFrame.LOCAL,
            )
            for cm in constraint_models if cm not in toremove]

    if actuation_model is not None:
        from example_parallel_robots.actuation_model import ActuationModel
        print('Reducing the actuation model')
        list_names = [model.names[idMot] for idMot in actuation_model.idMotJoints]
        reduced_actuation_model = ActuationModel(reduced_model,list_names)

    return(reduced_model, reduced_constraint_models, reduced_actuation_model, reduced_visual_model, reduced_collision_model)
            