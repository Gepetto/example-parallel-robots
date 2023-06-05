import numpy as np
class robot_actuation_model():
    """
    the actuation model of the robot
    """
    def __init__(self,model,names):
        self.motname=names
        self.nq=model.nq
        self.nv=model.nv
        self.getMotId_q(model)
        self.getFreeId_q(model)
        self.getMotId_v(model)
        self.getFreeId_v(model)

        
    def __str__(self):
        return(print("Id q motor: " + str(self.idqmot) + "\r" "Id v motor: " + str(self.idvmot) ))
    

    def getMotId_q(self,model):
        """
        GetMotId_q = (model, name_mot='mot')
        Return a list of ids corresponding to the configurations velocity associated with motors joints

        Arguments:
            model - robot model from pinocchio
            name_mot - string to be found in the motors joints names
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

    def getMotId_v(self,model):
        """
        GetMotId_q = (model, name_mot='mot')
        Return a list of ids corresponding to the configurations velocity associated with motors joints

        Arguments:
            model - robot model from pinocchio
            name_mot - string to be found in the motors joints names
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


    def getFreeId_q(self,model):
        
        Lidq=[]
        for i in range(model.nq):
            if not(i in self.idqmot):
                Lidq.append(i)
        self.idqfree=Lidq
        return(Lidq)
    
    def getFreeId_v(self,model):
        
        Lidv=[]
        for i in range(model.nv):
            if not(i in self.idvmot):
                Lidv.append(i)
        self.idvfree=Lidv
        return(Lidv)
    
    def qmot(self,q):
        qmot=[]
        for idq,i in enumerate(q):
            if idq in self.idqmot:
                qmot.append(i)
        return(np.array(qmot))
    
    def qfree(self,q):
        qfree=[]
        for idq,i in enumerate(q):
            if idq in self.idqfree:
                qfree.append(i)
        return(np.array(qfree))
    
    def vmot(self,v):
        vmot=[]
        for idv,i in enumerate(v):
            if idv in self.idvmot:
                vmot.append(i)
        return(np.array(vmot))
        
    def vfree(self,v):
        vfree=[]
        for idv,i in enumerate(v):
            if idv in self.idvfree:
                vfree.append(i)
        return(np.array(vfree))


    def completeq(self,qmot,qfree):
        q=np.zeros(self.nq)
        for i,idqmot in zip(qmot,self.idqmot):
            q[idqmot]=i

        for i,idqfree in zip(qfree,self.idqfree):
            q[idqmot]=i
        return(q)
    

    def completev(self,vmot,vfree):
        q=np.zeros(self.nv)
        for i,idqmot in zip(vmot,self.idvmot):
            q[idqmot]=i

        for i,idqfree in zip(vfree,self.idvfree):
            q[idqmot]=i
        return(q)
    
        
    