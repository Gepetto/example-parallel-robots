import unittest

from example_parallel_robots.loader_tools import load

# import os
# script_dir = os.path.dirname(__file__)
# mymodule_dir = os.path.join(script_dir, "../example_parallel_robots")
# sys.path.append(mymodule_dir)


class TestRobotLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.nameRobotToTest = [
            "5bar",
            "5bar3d",
            "5bar6d",
            "cassie_leg",
            "digit_leg",
            "digit_2legs",
            "disney_leg",
            "kangaroo_leg",
            "kangaroo_2legs",
            "delta",
            "talos_leg",
            "wl16_leg",
            "talos_full_closed",
            "talos_only_leg",
        ]

    def test_robot(self):
        for name in self.nameRobotToTest:
            load(name)


if __name__ == "__main__":
    unittest.main()
