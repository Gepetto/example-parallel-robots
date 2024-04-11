class FiveBarLinkage:
    path = "5bar_linkage"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class FiveBarLinkageIso3D:
    path = "5bar_linkage_iso3d"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class FiveBarLinkageIso6D:
    path = "5bar_linkage_iso6d"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class Cassie:
    path = "cassie_like"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class Digit:
    path = "digit_like"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class DigitBiped:
    path = "digit_like_2legs"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = True


class Disney:
    path = "disney_like"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class Kangaroo:
    path = "kangaroo_like"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False

class KangarooBiped:
    path = "kangaroo_like_2legs"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False

class Delta:
    path = "robot_delta"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


# class TalosClosed:
#     from .talos_closed import TalosClosed

<<<<<<< HEAD:example-parallel-robots/src/example_parallel_robots/robot_options.py
#     path = "talos_closed"
#     urdf_file = None
#     free_flyer = True
#     closed_loop = True
#     exec = TalosClosed
=======
#     path = "true_talos_2_legs"
#     urdf_file = "robot.urdf"
#     yaml_file = "robot.yaml"
#     free_flyer = True
>>>>>>> update_robot:example_parallel_robots/python/example_parallel_robots/robot_options.py


# class Talos:
#     from .talos_closed import TalosClosed

#     path = "talos_closed"
#     urdf_file = None
#     free_flyer = True
#     closed_loop = False
#     exec = TalosClosed


class TalosLeg:
    path = "talos_like"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


class TalosBiped:
    path = "true_talos_2_legs"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = True


class WL16:
    path = "wl16_like"
    urdf_file = "robot.urdf"
    yaml_file = "robot.yaml"
    free_flyer = False


ROBOTS = {
    "5bar": FiveBarLinkage,
    "5bar3d": FiveBarLinkageIso3D,
    "5bar6d": FiveBarLinkageIso6D,
    "cassie_leg": Cassie,
    "digit_leg": Digit,
    "digit_2legs": DigitBiped,
    "disney_leg": Disney,
    "kangaroo_leg": Kangaroo,
    "kangaroo_2legs": KangarooBiped,
    "delta": Delta,
    # "talos_full_closed": TalosClosed,
    # "talos_full_open": Talos,
    "talos_leg": TalosLeg,
    "talos_2legs": TalosBiped,
    "wl16_leg": WL16,
}
