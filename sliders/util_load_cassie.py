import pinocchio as pin
import hppfcl
import example_robot_data as robex
import numpy as np
from util_load_robots import defRootName,addXYZAxisToJoints,replaceGeomByXYZAxis,freeze,renameConstraints,addXYZAxisToConstraints

# List of joint string keys that should typically be locked
classic_cassie_blocker = [
    # *-ip and *-joint are the gearbox parallel models. 
    '-ip',
    '-roll-joint',
    '-yaw-joint',
    '-pitch-joint',
    '-knee-joint',
    '-foot-joint',

    # The two following joints are collocated to the rod constraints
    # hence useless.
    '-achilles-spring-joint',
    '-plantar-foot-joint',
]
# The following joints are used to model the flexibilities
# in the shin parallel actuation. 
cassie_spring_knee_joints = [
    '-knee-spring-joint',
    '-knee-shin-joint',
    '-shin-spring-joint',
    '-tarsus-spring-joint',
]
# The following constraints are used to close the loop on the gear-boxes
# They must be deactivated otherwise they would lock the motors.
classic_cassie_unecessary_constraints =\
    [
        '{pre}-roll-joint,{pre}-roll-op',
        '{pre}-yaw-joint,{pre}-yaw-op',
        '{pre}-pitch-joint,{pre}-pitch-op',
        '{pre}-knee-joint,{pre}-knee-op',
        '{pre}-foot-joint,{pre}-foot-op',
    ]
# See https://stackoverflow.com/questions/42497625/how-to-postpone-defer-the-evaluation-of-f-strings
# for understanding the eval(...) syntax
classic_cassie_unecessary_constraints = \
    [ eval(f"f'{c}'") for c in classic_cassie_unecessary_constraints for pre in [ 'right', 'left'] ]

def fixCassieConstraints(cassie,verbose=False):
    '''
    Some constraints are unproperly define (why?). This hack fixes them. But we should 
    understand why and make a better fix.
    '''

    for prefix in ['right', 'left']:
        # First invert the two joints around the tarsus crank:
        # - the first should be a sphere, the second a revolute.
        i = cassie.model.getJointId(f'{prefix}-crank-rod-joint')
        idx_q,idx_v = cassie.model.joints[i].idx_q,cassie.model.joints[i].idx_v
        cassie.model.joints[i] = pin.JointModelSpherical()
        cassie.model.joints[i].setIndexes(i,idx_q,idx_v)
        j1 = cassie.model.joints[i]
        ax1 = cassie.data.joints[j1.id].S[3:]
        if verbose:
            print(f'New top-rod {prefix} joint:',j1)
    
        i = cassie.model.getJointId(f'{prefix}-plantar-foot-joint')
        cassie.model.joints[i] = pin.JointModelRZ()
        cassie.model.joints[i].setIndexes(i,idx_q+4,idx_v+3)
        cassie.data = cassie.model.createData()
        j2 = cassie.model.joints[i]
        if verbose:
            print(f'New bottom-rod {prefix} joint:',j2)
    
        for nq in cassie.model.referenceConfigurations:
            # Reorder the joint in the configuration. What follows is an ad-hoc tentative
            # to automatically find a configuration respecting the same constraint as the
            # previous joint organisation. Not fully clean and succesfull, but good enough
            n,q = nq.key(),nq.data().copy()
            rot = pin.AngleAxis(q[j1.idx_q],ax1)
            quat = pin.Quaternion(rot.matrix())
            cassie.model.referenceConfigurations[n][j1.idx_q:j1.idx_q+j1.nq] = quat.coeffs()
            cassie.model.referenceConfigurations[n][j2.idx_q] = 0
        
    for cm in (cm for cm in cassie.constraint_models
               if 'achilles-spring-joint' in cassie.model.names[cm.joint1_id]
               and 'tarsus-spring-joint' in cassie.model.names[cm.joint2_id]):
        print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()  # M2 is Id, so inverse not needed, but I have the intuition it is more generic.
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D
    
    for cm in (cm for cm in cassie.constraint_models
               if 'shin-spring-joint' in cassie.model.names[cm.joint1_id]
               and 'right-knee-spring-joint' in cassie.model.names[cm.joint2_id]):
        print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D

    for cm in (cm for cm in cassie.constraint_models
               if 'plantar-foot-joint' in cassie.model.names[cm.joint1_id]
               and 'foot-op' in cassie.model.names[cm.joint2_id]):
        print(f'Fix constraint {cm.name} ({cm})')
        M1 = cm.joint1_placement.copy()
        M2 = cm.joint2_placement.copy()
        cm.joint1_placement = M2.inverse()
        cm.joint2_placement = M1.inverse()
        cm.type = pin.ContactType.CONTACT_3D


def loadCassieAndFixIt(initViewer=False):
    '''
    Load Cassie from example robot data, then fix the model errors, 
    freeze the unecessary joints, and load the model in the viewer.
    '''

    cassie=robex.load('cassie')
    fixCassieConstraints(cassie)
    defRootName(cassie.model,'cassie')
    renameConstraints(cassie)
    cassie.constraint_models = [ cm for cm in cassie.constraint_models
                                 if cm.name not in classic_cassie_unecessary_constraints ]
    addXYZAxisToConstraints(cassie.model,cassie.visual_model,cassie.constraint_models)
    jointsToLock = []
    for key in classic_cassie_blocker+cassie_spring_knee_joints+['left']:
        jointsToLock += [ i for i,n in enumerate(cassie.model.names) if key in n]
    # Freeze expect a list of uniq identifiers, so use list(set()) to enforce that
    freeze(cassie,list(set(jointsToLock)),'standing',rebuildData=False)
    addXYZAxisToJoints(cassie.model,cassie.visual_model)
    cassie.rebuildData()
    if initViewer:
        cassie.initViewer(loadModel=True)
        replaceGeomByXYZAxis(cassie.visual_model,cassie.viz)
        cassie.display(cassie.q0)
    return cassie
