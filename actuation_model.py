import numpy as np
class robot_actuation_model():
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
    def __init__(self,model,names):
        self.motname=names
        self.nq=model.nq
        self.nv=model.nv
        self.__getMotId_q__(model)
        self.__getFreeId_q__(model)
        self.__getMotId_v__(model)
        self.__getFreeId_v__(model)

        
    def __str__(self):
        return(print("Id q motor: " + str(self.idqmot) + "\r" "Id v motor: " + str(self.idvmot) ))
    

    def __getMotId_q__(self,model):
        """
        GetMotId_q = (model)
        Return a list of ids corresponding to the configurations velocity associated with motors joints

        Arguments:
            model - robot model from pinocchio
        Return:
            Lid - List of motors configuration velocity ids
        """
        Lidq = []
        for i, name in enumerate(model.names):
            if name in self.motname:
                idq=model.joints[i].idx_q
                nq=model.joints[i].nq
                for j in range(nq):
                    Lidq.append(idq+j)
        self.idqmot=Lidq
        return Lidq

    def __getMotId_v__(self,model):
        """
        GetMotId_q = (model)
        Return a list of ids corresponding to the configurations velocity associated with motors joints

        Arguments:
            model - robot model from pinocchio
        Return:
            Lid - List of motors configuration velocity ids
        """
        Lidv = []
        for i, name in enumerate(model.names):
            if name in self.motname:
                idv=model.joints[i].idx_v
                nv=model.joints[i].nv
                for j in range(nv):
                    Lidv.append(idv+j)
        self.idvmot=Lidv
        return Lidv


    def __getFreeId_q__(self,model):
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
    
    def __getFreeId_v__(self,model):
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
        return(Lidv)
    
    def qmot(self,q):
        """
        qmot = (q)
        return the configuration vector associatet d to the motor coordinate

        argument:
            q - the complete configuration vector
        return :
            qmot  - the motor configuration vector
        """
        qmot=[]
        for idq,i in enumerate(q):
            if idq in self.idqmot:
                qmot.append(i)
        return(np.array(qmot))
    
    def qfree(self,q):
        """
        qfree = (q)
        return the configuration vector associatted d to the free coordinate

        argument:
            q - the complete configuration vector
        return :
            qfree  - the free configuration vector
        """
        qfree=[]
        for idq,i in enumerate(q):
            if idq in self.idqfree:
                qfree.append(i)
        return(np.array(qfree))
    
    def vmot(self,v):
        """
        vmot = (v)
        return the configuration vector associatted d to the motor coordinate

        argument:
            v - the complete configuration velocity vector
        return :
            vmot  - the motor configuration velocity vector
        """
        vmot=[]
        for idv,i in enumerate(v):
            if idv in self.idvmot:
                vmot.append(i)
        return(np.array(vmot))
        
    def vfree(self,v):
        """
        vfree = (v)
        return the configuration velocity vector associatted d to the free coordinate

        argument:
            v - the complete configuration velocity vector
        return :
            vfree  - the free configuration velocity vector
        """
        vfree=[]
        for idv,i in enumerate(v):
            if idv in self.idvfree:
                vfree.append(i)
        return(np.array(vfree))


    def completeq(self,qmot,qfree):
        """
        completeq = (qmot,qfree)
        concatenate qmot qfree in respect with motor and free id
        """
        q=np.zeros(self.nq)
        for i,idqmot in zip(qmot,self.idqmot):
            q[idqmot]=i

        for i,idqfree in zip(qfree,self.idqfree):
            q[idqmot]=i
        return(q)
    

    def completev(self,vmot,vfree):
        """
        completev = (vmot,vfree)
        concatenate vmot vfree in respect with motor and free id
        """
        q=np.zeros(self.nv)
        for i,idqmot in zip(vmot,self.idvmot):
            q[idqmot]=i

        for i,idqfree in zip(vfree,self.idvfree):
            q[idqmot]=i
        return(q)
    
        
    