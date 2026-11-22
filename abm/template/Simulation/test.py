from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm, LinearGMRESSolver, ImplicitSourceTerm
from variablevals import *

def tester(cellpresente, cellpresentndn, cellpresentna, cellpresentm1, cellpresentm2, cytokines, mesh):
    '''
    SOlves the diffusion equations with a mesh into consideration
    '''
    valueboundaries = 0.
    # Define the center of the mesh
    X, Y = mesh.faceCenters
    # Define the boundaries of the domain
    boundariesall = ((mesh.facesBottom & (X < boundaryat))|(mesh.facesTop & (X > nx-boundaryat)) | (mesh.facesRight & (Y > ny-boundaryat)) | (mesh.facesLeft & (Y < boundaryat)))

    # Define the Variables being solved
    il8 = cytokines[0]
    il1 = cytokines[1]
    il6 = cytokines[2]
    il10 = cytokines[3]
    tnf = cytokines[4]
    tgf = cytokines[5]
    
    il8 = CellVariable(mesh = mesh,value= il8)
    il1 = CellVariable(mesh = mesh,value= il1)
    il6 = CellVariable(mesh = mesh,value= il6)
    il10 = CellVariable(mesh = mesh,value= il10)
    tnf = CellVariable(mesh = mesh,value= tnf)
    tgf = CellVariable(mesh = mesh,value= tgf)
    
    # Constrain the variables within bounds
    il8.constrain(valueboundaries, boundariesall)
    il1.constrain(valueboundaries, boundariesall)
    il6.constrain(valueboundaries, boundariesall)
    il10.constrain(valueboundaries, boundariesall)
    tnf.constrain(valueboundaries, boundariesall)
    tgf.constrain(valueboundaries, boundariesall)
    
    # Define thee solver used
    mysolver=LinearGMRESSolver()

    # Define the equations
    eqil8 = TransientTerm() == DiffusionTerm(coeff=Dil8) - ImplicitSourceTerm(muil8) + keil8*cellpresente + kndnil8*cellpresentndn - ImplicitSourceTerm(thetanail8*cellpresentna)
    eqil1 = TransientTerm() == DiffusionTerm(coeff=Dil1) - ImplicitSourceTerm(muil1) + knail1*cellpresentna
    eqil6 = TransientTerm() == DiffusionTerm(coeff=Dil6) - ImplicitSourceTerm(muil6) + km1il6*cellpresentm1
    eqil10 = TransientTerm() == DiffusionTerm(coeff=Dil10) - ImplicitSourceTerm(muil10) + km2il10*cellpresentm1
    eqtnf = TransientTerm() == DiffusionTerm(coeff=Dtnf) - ImplicitSourceTerm(mutnf) + knatnf*cellpresentna + km1tnf*cellpresentm1
    eqtgf = TransientTerm() == DiffusionTerm(coeff=Dtgf) - ImplicitSourceTerm(mutgf) + km2tgf*cellpresentm2

    # Solve for the duration(dt)
    for i in range(fipy_duration):
        eqil8.solve(var= il8, dt=1.0,solver=mysolver)
        eqil1.solve(var= il1, dt=1.0,solver=mysolver)
        eqil6.solve(var= il6, dt=1.0,solver=mysolver)
        eqil10.solve(var= il10, dt=1.0,solver=mysolver)
        eqtnf.solve(var= tnf, dt=1.0,solver=mysolver)
        eqtgf.solve(var= tgf, dt=1.0,solver=mysolver)
    return (il8,il1,il6,il10,tnf,tgf)



