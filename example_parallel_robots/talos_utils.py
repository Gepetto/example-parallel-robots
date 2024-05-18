# import numpy as np
# import example_robot_data as robex
# import pinocchio as pin
# import meshcat
# from pinocchio.visualize import MeshcatVisualizer
# import hppfcl
# import re
# from .actuation_model import ActuationModel
# from .freeze_joints import freezeJoints
# from .loader_tools import load


# def onlyleg():
#     model, constraint_models, actuation_model, visual_model, collision_model = load("talos_full_closed")
#     names_joints_to_lock = [
#             # "universe",
#             # "root_joint",
#             "mot_arm_left_1_joint",
#             "mot_arm_left_2_joint",
#             "mot_arm_left_3_joint",
#             # "arm_left_4_joint",
#             "mot_arm_left_5_joint",
#             "mot_arm_left_6_joint",
#             "mot_arm_left_7_joint",
#             "mot_arm_right_1_joint",
#             "mot_arm_right_2_joint",
#             "mot_arm_right_3_joint",
#             # "arm_right_4_joint",
#             "mot_arm_right_5_joint",
#             "mot_arm_right_6_joint",
#             "mot_arm_right_7_joint",
#             "mot_gripper_left_joint",
#             "mot_gripper_right_joint",
#             "mot_head_1_joint",
#             "mot_head_2_joint",
#             "mot_torso_1_joint",
#             "mot_torso_2_joint",
#         ]
#     ids_joints_to_lock = [
#         i for (i, n) in enumerate(new_model.names) if n in names_joints_to_lock
#     ]
#     q0=pin.neutral(model)
#     (
#         new_model,
#         constraint_models,
#         actuation_model,
#         visual_model,
#         collision_model,
#     ) = freezeJoints(
#         new_model,
#         constraint_models,
#         actuation_model,
#         visual_model,
#         collision_model,
#         ids_joints_to_lock,
#         q0,
#     )
#     return (
#         new_model,
#         constraint_models,
#         actuation_model,
#         visual_model,
#         collision_model,
#     )

# if __name__ == "__main__":
#     model, cm, am, visual_model, collision_model = TalosClosed(
#         closed_loop=True, only_legs=True
#     )
#     # new_model, geometry_models, cm, am = reorganizeModels(model, [visual_model, collision_model], cm)
#     # new_visual_model, new_collision_model = geometry_models[0], geometry_models[1]
#     #
#     q0 = pin.neutral(model)
#     viz = MeshcatVisualizer(model, visual_model, visual_model)
#     viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
#     viz.clean()
#     viz.loadViewerModel(rootNodeName="universe")
#     viz.display(q0)
#     #
#     # input()
#     # q0 = pin.neutral(new_model)
#     # viz = MeshcatVisualizer(new_model, new_visual_model, new_visual_model)
#     # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
#     # viz.clean()
#     # viz.loadViewerModel(rootNodeName="universe")
#     # viz.display(q0)
