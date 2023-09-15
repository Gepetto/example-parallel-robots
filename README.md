# General information
This repository is a WIP of a library that compiles tools for closed loop kinematics modelization. It contains formulation of resolution of several kinematic problems on parallel robots.

# Requirement
* **Pinocchio 3.x with optionnal Casadi support**
* Casadi and IpOpt [Optionnal]
* Scipy [is Casadi is not installed]
* Gepetto Viewer or Meshcat (codes are curretly written for meshcat but can be adapted to Gepetto Viewer)
* Standard Python librairies (Numpy, yaml, re)

# Contents
## Robot modeling tools
### How we model closed loops 
In this repo, we propose to use a representation of closed loop kinematics in robots as open looped chains with inner contact constraints. The actuation information (e.g. which joints are actuated) are stored in a object of type *ActuationModel*. This information needed to stored a robot model is contained in a URDF file and a YAML file.
### How to load models
We propose a code that allows the user to load a robot model. The function *complete robot loader* take the path to the folder containing the two files and returns Pinocchio object and an ActuationModel object describing the robot. Use example can be found in the test scripts (see below).

## Kinematic problem
We are solving in this repo several problems a the special case of robots containing closed kinematic loops. 
* *closed_loop_mount.py* contains methods to solve the mounting problem - i.e. finding a robot configuration that satisfies all the contraints. 
To test this code, the script *tests_mounting.py* can be ran, after running *meshcat_server* in terminal. It should load a robot model and display its neutral and mounted configuration.
* *closed_loop_kinematics* presents method to solve geometry problem - i.e. involving only the joints positions and not velocities. This includes forward geometry and inverse geometry.
To test this code, one can run the scripts *test_forwardKinematics.py* and *test_inverseKinematics.py*. It should respectivelly change the robot configuration according to the given actuators positions and change the robot configuration so that the position of the effecteur matches a target position.
* *closed_loop_jacobian* contains methods related to the robot Jacobian and with velocity kinematic problems.

## Robot models
This repository also serves as a storage of robot models. Therefore, in the folder robots, several models of parallel robots are stored and they can easily be used following the above instructions.
