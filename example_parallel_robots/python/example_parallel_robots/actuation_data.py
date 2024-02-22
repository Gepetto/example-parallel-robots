import pinocchio as pin
import numpy as np

class ActuationData():
    """
    Defines the actuation data of a robot
    robot_actuation_data = ActuationData(model, constraints_models, actuation_model)
    Arguments:
        model - robot model
        constraints_models - List of the constraint model associated to the robot
        actuation_model - Robot actuation model
    Attributes:
        TO DO
    Methodes:
        None
    
    """
    def __init__(self, model, constraints_models,actuation_model):

        Lidmot=actuation_model.mot_ids_v
        free_ids_v=actuation_model.free_ids_v
        nv=model.nv
        nv_mot=actuation_model.nv_mot
        nv_free=actuation_model.nv_free
        nc=0
        for c in constraints_models:
            nc+=c.size()

        ## init different matrix
        self.Smot=np.zeros((nv,nv_mot))
        self.Sfree=np.zeros((nv,nv_free))
        self.Jmot=np.zeros((nc,nv_mot))
        self.Jfree=np.zeros((nc,nv_free))
        self.Mmot=np.zeros((nv_mot,nv_mot))
        self.dq=np.zeros((nv,nv_mot))

        #PRIVATE
        self.dq_no=np.zeros((nv,nv_mot))
    
        #init a list of constraint_jacobian
        self.LJ=[np.array(())]*len(constraints_models)
        constraint_data=[c.createData() for c in constraints_models]
        data=model.createData()
        for (cm,cd,i) in zip(constraints_models,constraint_data,range(len(self.LJ))):
            self.LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)


        #selection matrix for actuated parallel robot
        self.Smot[:,:]=0
        self.Smot[Lidmot,range(nv_mot)]=1
        self.Sfree[:,:]=0
        self.Sfree[free_ids_v,range(nv_free)]=1


        # to delete
        self.Jf_closed=pin.computeFrameJacobian(model,model.createData(),np.zeros(model.nq),0,pin.LOCAL)@self.dq


        # init of different size of vector
        self.vq=np.zeros(nv)
        self.vqmot=np.zeros(nv_mot-6)
        self.vqfree=np.zeros(nv_free)
        self.vqmotfree=np.zeros(nv-6)

        #list of constraint type
        self.constraints_sizes=[J.shape[0] for J in self.LJ]


        #to delete ?
        self.pinvJfree=np.linalg.pinv(self.Jfree)