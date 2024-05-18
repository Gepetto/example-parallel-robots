import pinocchio as pin


def reorganizeModelDepthFirst(model):
    """
    Reorganizes the model by creating a new model and propagating the joints and bodies.
    Args:
        model (pin.Model): The model to be reorganized.
    Returns:
        pin.Model: The reorganized model.
    """

    def propagate(stack, new_model, i):
        if len(stack) == 0:
            return new_model
        if i == 1000:
            raise (
                RecursionError(
                    "Reached max recursion depth when reorganizing the model"
                )
            )
        else:
            (jointId, parentId) = stack.pop()
            jId = new_model.addJoint(
                parentId,  #
                model.joints[jointId],  #
                model.jointPlacements[jointId],  #
                model.names[jointId],
            )  #
            new_model.appendBodyToJoint(
                jId,
                model.inertias[jointId],
                pin.SE3.Identity(),  #
            )  #
            children = model.children[jointId]
            for c in children:
                stack.append((c, jId))
        propagate(stack, new_model, i + 1)

    new_model = pin.Model()
    new_model.name = model.name
    propagate([(1, 0)], new_model, 0)
    return new_model


def reorganizeModels(old_model, old_geometry_models=[], old_constraint_models=[]):
    """
    Reorganizes the models by creating a new model, updating frames, geometry models, and constraint models.

    Args:
        old_model (pin.Model): The old model to be reorganized.
        old_geometry_models (list): List of old geometry models to be reorganized, default to empty.
        old_constraint_models (list): List of old constraint models to be reorganized, default to empty.

    Returns:
        tuple: A tuple containing the reorganized model, geometry models, and constraint models.
    """
    # Model
    model = reorganizeModelDepthFirst(old_model)
    # Frames
    for frame in old_model.frames[1:]:
        name = frame.name
        parent_joint = model.getJointId(
            old_model.names[frame.parentJoint]
        )  # Should be a joint Id
        placement = frame.placement
        frame = pin.Frame(name, parent_joint, placement, pin.BODY)
        model.addFrame(frame, False)
    # Geometry model
    geometry_models = []
    for old_geom_model in old_geometry_models:
        geom_model = old_geom_model.copy()
        for gm in geom_model.geometryObjects:
            gm.parentJoint = model.getJointId(old_model.names[gm.parentJoint])
            gm.parentFrame = model.getFrameId(old_model.frames[gm.parentFrame].name)
        geometry_models.append(geom_model)
    # Constraint models
    constraint_models = []
    for cm in old_constraint_models:
        constraint_models.append(
            pin.RigidConstraintModel(
                cm.type,
                model,
                model.getJointId(old_model.names[cm.joint1_id]),
                cm.joint1_placement,
                model.getJointId(old_model.names[cm.joint2_id]),
                cm.joint2_placement,
                pin.ReferenceFrame.LOCAL,
            )
        )
    return (model, geometry_models, constraint_models)
