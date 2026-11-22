from cc3d import CompuCellSetup
from variablevals import relaxationmcs      
from endothelialSteppables import endothelialSteppable

CompuCellSetup.register_steppable(steppable=endothelialSteppable(frequency=int(relaxationmcs)))



        
from endothelialSteppables import celldivisionSteppable
CompuCellSetup.register_steppable(steppable=celldivisionSteppable(frequency=int(relaxationmcs)))


CompuCellSetup.run()
