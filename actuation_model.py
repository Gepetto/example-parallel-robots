import numpy as np
class ActuationModel():
    """
    the actuation model of the robot,
    robot_actuation_model(model,names)
    argument :
        model - robot model
        names - list of the name of motor joint name
    contain :
        self.nq, self.nv size of configuration/velocity space
        self.idqmot , self.idvmot the id of the motor joint inside a configuration / velocity vector
        self.idfree, self.idvfree the id of the free joint inside a configuration / velocity vector
    
    """
    def __init__(self, model, names):
        self.motnames = names
        self.model = model
        self.idMotJoints = []
        self.getMotId_q(model)
        self.getFreeId_q(model)
        self.getMotId_v(model)
        self.getFreeId_v(model)

        
    def __str__(self):
        return(print("Id q motor: " + str(self.idqmot) + "\r" "Id v motor: " + str(self.idvmot) ))
    

    def getMotId_q(self):
        """
        GetMotId_q = (model)
        Return a list of ids corresponding to the configurations velocity associated with motors joints

        Arguments:
            model - robot model from pinocchio
        Return:
            Lid - List of motors configuration velocity ids
        """
        Lidq = []
        for i, name in enumerate(self.model.names):
            for motname in self.motnames:
                if motname in name:
                    self.idMotJoints.append(i)
                    idq = self.model.joints[i].idx_q
                    nq = self.model.joints[i].nq
                    for j in range(nq):
                        Lidq.append(idq+j)
        self.idqmot=Lidq

    def getMotId_v(self):
        """
        GetMotId_q = (model)
        Return a list of ids corresponding to the configurations velocity associated with motors joints

        Arguments:
            model - robot model from pinocchio
        Return:
            Lid - List of motors configuration velocity ids
        """
        Lidv = []
        for i, name in enumerate(self.model.names):
            for motname in self.motnames:
                if motname in name:
                    idv = self.model.joints[i].idx_v
                    nv = self.model.joints[i].nv
                    for j in range(nv):
                        Lidv.append(idv+j)
        self.idvmot=Lidv

    def getFreeId_q(self,model):
        """
        GetFreeId_q = (model)
        Return a list of ids corresponding to the configurations vector associated with motors joints

        Arguments:
            model - robot model from pinocchio
        Return:
            Lid - List of motors configuration velocity ids
        """
        Lidq=[]
        for i in range(model.nq):
            if not(i in self.idqmot):
                Lidq.append(i)
        self.idqfree=Lidq
        return(Lidq)
    
    def getFreeId_v(self,model):
        """
        GetFreeId_v = (model)
        Return a list of ids corresponding to the configurations velocity vector associated with motors joints

        Arguments:
            model - robot model from pinocchio
        Return:
            Lid - List of motors configuration velocity ids
        """
        Lidv=[]
        for i in range(model.nv):
            if not(i in self.idvmot):
                Lidv.append(i)
        self.idvfree=Lidv