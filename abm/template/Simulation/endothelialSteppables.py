from cc3d.core.PySteppables import *
from fipy import CellVariable, Grid2D, dump
from test import tester
import variablevals as vv
import gc
import os
import sys
import random
import time
global mesh
from numpy import *
import numpy as np
from builtins import range
global cytokines
global cellpresent
global fullFileName
import csv


# globals for search
global setLambda
global setSaturationCoef

# sigmoid predefinitions
# sigmoida = {{sigmoida}}
# sigmoidb = {{sigmoidb}}

# Predefinition of parameters
setlambda = 2000 #{{setLambda}}
setSaturationCoef = 10**-11#{{setSaturationCoef}}
endocount = {{endocount}} #{{endocount}}

def randopos(xboundary, yboundary, typein):
    '''
    Function randomization of position within lattice
    '''
    if typein!=1:
        randxhi = random.randint(vv.nx-xboundary+2, vv.nx-2)
        randxlo = random.randint(2, xboundary-2)
        randyall = random.randint(2, vv.ny-2)
        randyhi = random.randint(vv.ny-yboundary+2, vv.ny-2)
        randylo = random.randint(2, yboundary-2)
        randxall = random.randint(2, vv.nx-2)
        hixvals = [randxhi, randyall]
        loxvals = [randxlo, randyall]
        hiyvals = [randxall, randyhi]
        loyvals = [randxall, randylo]
    if typein == 1:
        randxhi = random.randint(xboundary+2, vv.nx-2-xboundary)
        randxlo = random.randint(xboundary+2, vv.nx-2-xboundary)
        randyall = random.randint(yboundary+2, vv.ny-2-yboundary)
        randyhi = random.randint(yboundary+2, vv.ny-2-yboundary)
        randylo = random.randint(yboundary+2, vv.ny-2-yboundary)
        randxall = random.randint(xboundary+2, vv.nx-2-xboundary)
        hixvals = [randxhi, randyall]
        loxvals = [randxlo, randyall]
        hiyvals = [randxall, randyhi]
        loyvals = [randxall, randylo]
    rands = [hixvals, loxvals, hiyvals, loyvals]
    randselect = random.randint(0,4)
    randx = rands[randselect][0]
    randy = rands[randselect][1]
    return [randx, randy]
    
def sigmoid(x, a, b):
    '''
    Sigmoid function:
    Returns the curve for,
    a:first inflection point
    b:second inflection point
    over x(time)
    '''
    z = np.exp(-a*(x - b))
    sig = 1 / (1 + z)

    return sig
    
def modmm(x, km, vmax):
    '''
    Modular Michaelis Menten function:
    Returns the curve for,
    vmax:first inflection point
    km:second inflection point
    over x(time)
    '''
    mm = abs((vmax * x)/(km + x))
    
    return mm
# Get the dimensions of the lattice
nx = vv.nx
ny = nx
dx = vv.dx
dy = dx
L = dx * nx

# Construct the lattice in a 2D mesh
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# Check if the solution variable is in the mesh - endothelial cell
cellpresente = CellVariable(name = "solution variable",
           mesh = mesh,
           value = 0)
# Check if the solution variable is in the mesh - ND Neutrophil    
cellpresentndn = CellVariable(name = "solution variable",
           mesh = mesh,
           value = 0)
# Check if the solution variable is in the mesh - Activated neutrophil
cellpresentna = CellVariable(name = "solution variable",
           mesh = mesh,
           value = 0)
# Check if the solution variable is in the mesh - Macrophage 1
cellpresentm1 = CellVariable(name = "solution variable",
           mesh = mesh,
           value = 0)
# Check if the solution variable is in the mesh - Macrophage 2           
cellpresentm2 = CellVariable(name = "solution variable",
           mesh = mesh,
           value = 0)      
# For every solution variable disperse cytokines            
cytokines = [CellVariable(name = "solution variable", mesh = mesh, value = 0.0) for i in range(vv.total_cytokines)]

class endothelialSteppable(SteppableBasePy):

    def __init__(self,frequency=int(vv.relaxationmcs)):
        SteppableBasePy.__init__(self,frequency)
     

    def start(self):
        '''
        Initialization of simulation. Creates the field for cytokines 
        and distribution of cells, with the settings.
        Also starts the plotting settings.
        '''
        
        global fullFileName   
        global setSaturationCoef
        global endocount
        
        # Create the cytokine scalar field
        self.scalarFieldil8=self.create_scalar_field_py("il8")
        self.scalarFieldil1=self.create_scalar_field_py("il1")
        self.scalarFieldil6=self.create_scalar_field_py("il6")
        self.scalarFieldil10=self.create_scalar_field_py("il10")
        self.scalarFieldtnf=self.create_scalar_field_py("tnf")
        self.scalarFieldtgf=self.create_scalar_field_py("tgf")
        
        # Create text file to store the output for cell counts
        fileHandle,fullFileName=self.open_file_in_simulation_output_folder("datafiles/creatdoc.txt","w")
        if not os.path.exists(os.path.dirname(fullFileName)):
            os.makedirs(os.path.dirname(fullFileName))
        # Initialize Plot cell counts
        self.plot_win = self.add_new_plot_window(title='Cell counts',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Cell types', x_scale_type='linear', y_scale_type='linear',
                                                 grid=False)
        labels = ['Endothelial', 'Neutrophils', 'Monocytes', 'Fibroblast', 'Neutrophil A', 'ND Neutrophil', 'Monocyte R', 'Macrophage I', 'Macrophage II', 'Myofibroblast']
        colors = ["blue", "brown", "cyan", "violet", "red", "pink", "yellow", "orange", "darkblue", "green"]
        [self.plot_win.add_plot(labels[i], style='Lines', color=colors[i]) for i in range(len(colors))]
        
        
        
        
        """
        any code in the start function runs before MCS=0
        """
        
        # size of cell will be 3x3x1
        # Distribution of cells in the lattice
        for i in range(0,endocount):
            randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
            self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.ENDOTHELIAL)
            
        for i in range(0,1000):
            randx,randy = randopos(vv.boundaryat, vv.boundaryat, 2)
            self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.NEUTROPHIL)
            
        for i in range(0,100):
            randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
            self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.NEUTROPHIL)
            
        for i in range(0,100):
            randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
            self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.FIBROBLAST)
            
        for i in range(0,25):
            randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
            self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.MYOFIBROBLAST)
        
        for i in range(0,1000):
            randx,randy = randopos(vv.boundaryat, vv.boundaryat, 3)
            self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.MONOCYTE)
            
        # All cells must have a volume and a range of movement
        for cell in self.cell_list:
            cell.targetVolume = 1
            cell.lambdaVolume = 10.0
            
            # Endothelial cell settings
            if cell.type == 1: # Endothelial cell
                cell.dict["life"] = 0
                cell.dict["span"] = vv.lifespane
                cell.dict["divide"] = vv.divpre
                cell.dict["dividepr"] = 1
                
            # Neutrophil settings
            if cell.type == 2: # Neutrophil 
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "IL8")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = vv.lifespannr
                cell.dict["dividepr"] = vv.divprnr
                cell.dict["life"] = random.randint(0,vv.lifespannr)
                
            # Monocyte settings for chemotaxis towards IL1
            if cell.type == 3: # Monocyte
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "IL1")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = vv.lifespanm
                cell.dict["dividepr"] = vv.divprm
                cell.dict["life"] = random.randint(0,vv.lifespanm)
                
            # Monocyte settings for chemotaxis towards TNFalpha
            if cell.type == 3: # Monocyte
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "TNF")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = vv.lifespanm
                cell.dict["dividepr"] = vv.divprm
                cell.dict["life"] = random.randint(0,vv.lifespanm)
                
            # Fibroblast settings for chemotaxis    
            if cell.type == 4: # Fibroblast
                cd = self.chemotaxisPlugin.addChemotaxisData(cell, "TGF")
                cd.setLambda(setlambda)
                cd.setChemotactTowards("MEDIUM")
                cd.setSaturationCoef(setSaturationCoef)
                cell.dict["span"] = vv.lifespanf
                cell.dict["divide"] = random.randint(0,vv.divprf)
                cell.dict["dividepr"] = vv.divprf
                cell.dict["life"] = random.randint(0,vv.lifespanf)
                
            # Myofibroblast settings  
            if cell.type == 10:
                cell.dict["span"] = vv.lifespanf
                cell.dict["divide"] = random.randint(0,vv.divprf)
                cell.dict["dividepr"] = vv.divprf
                cell.dict["life"] = random.randint(0,vv.lifespanf)
                


    def step(self,mcs):
        '''
        Function defines what is done for each MCS step
        '''
        global cytokines
        global cellpresent
        global fullFileName
        global relaxationmcs
 
        ccount = np.zeros(vv.total_celltypes+1)
        # Check the mcs remaining, if mcs = 0 then
        if mcs%(vv.relaxationmcs*10)==0:
            # Randomize a neutrophil placement number in the cell field 
            for i in range(0,10):
                randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
                self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.NEUTROPHIL)
            # Randomize a fibroblast placement number in the cell field     
            for i in range(0,10):
                randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
                self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.FIBROBLAST)
            # Randomize a myofibroblast placement number in the cell field     
            for i in range(0,3):
                randx,randy = randopos(vv.boundaryat, vv.boundaryat, 1)
                self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.MYOFIBROBLAST)
            # Randomize a monocyte placement number in the cell field     
            for i in range(0,100):
                randx,randy = randopos(vv.boundaryat, vv.boundaryat, 3)
                self.cell_field[randx:randx+1, randy:randy+1, 0] = self.new_cell(self.MONOCYTE)
            # All cells must get an update on the volume and a range of movement, but also update each cell settings
            for cell in self.cell_list:
                cell.targetVolume = 1
                cell.lambdaVolume = 10.0
                
                # Check cell type and update its characteristics
                if cell.type == 1: # Endothelial
                    cell.dict["life"] = 0
                    cell.dict["span"] = vv.lifespane
                    cell.dict["divide"] = vv.divpre
                    cell.dict["dividepr"] = 1
                    
                # Check cell type and update its characteristics   
                if cell.type == 2: # Neutrophil
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "IL8")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = vv.lifespannr
                    cell.dict["dividepr"] = vv.divprnr
                    cell.dict["life"] = random.randint(0,vv.lifespannr)
                    
                # Check cell type and update its characteristics    
                if cell.type == 3: # Monocyte
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "IL1")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = vv.lifespanm
                    cell.dict["dividepr"] = vv.divprm
                    cell.dict["life"] = random.randint(0,vv.lifespanm)
                    
                # Check cell type and update its characteristics    
                if cell.type == 3: # Monocyte
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "TNF")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = vv.lifespanm
                    cell.dict["dividepr"] = vv.divprm
                    cell.dict["life"] = random.randint(0,vv.lifespanm)
                    
                # Check cell type and update its characteristics    
                if cell.type == 4: # Fibroblast
                    cd = self.chemotaxisPlugin.addChemotaxisData(cell, "TGF")
                    cd.setLambda(setlambda)
                    cd.setChemotactTowards("MEDIUM")
                    cd.setSaturationCoef(setSaturationCoef)
                    cell.dict["span"] = vv.lifespanf
                    cell.dict["divide"] = random.randint(0,vv.divprf)
                    cell.dict["dividepr"] = vv.divprf
                    cell.dict["life"] = random.randint(0,vv.lifespanf)
                    
                # Check cell type and update its characteristics
                if cell.type == 10: # Myofibroblast
                    cell.dict["span"] = vv.lifespanf
                    cell.dict["divide"] = random.randint(0,vv.divprf)
                    cell.dict["dividepr"] = vv.divprf
                    cell.dict["life"] = random.randint(0,vv.lifespanf)
               
        # For all cell types
        for cell in self.cell_list:
            # Save their location on the lattice
            xCOM = cell.xCOM
            yCOM = cell.yCOM
            zCOM = cell.zCOM
            # within bounds
            if xCOM>=500:
                xCOM = 499
            if yCOM>=500:
                yCOM = 499
            # if zCOM>=500:
                # zCOMC = 499
                
            # for the positioned within bounds
            pos=int(xCOM)+(int(yCOM))*nx
            if pos>250000:
                print(xCOM, yCOM)
                
            # the cells live    
            cell.dict["life"] += 1
            # if certain cell is present at certain position, then its value is 1
            if cell.type == 1:
                cellpresente[pos]=1.  
            # if certain cell is present at certain position, then its value is 1   
            if cell.type == 6:
                cellpresentndn[pos]=1.
            # if certain cell is present at certain position, then its value is 1    
            if cell.type == 5:
                cellpresentna[pos]=1.
            # if certain cell is present at certain position, then its value is 1    
            if cell.type == 8:
                cellpresentm1[pos]=1.
            # if certain cell is present at certain position, then its value is 1    
            if cell.type == 9:
                cellpresentm2[pos]=1.
        # Simulate the cytokines and cells  with the current positional arguments in the current state of the mesh        
        cytokines = tester(cellpresente,cellpresentndn,cellpresentna,cellpresentm1,cellpresentm2,cytokines, mesh)
        
        # Create the scalr field to place the cytokines
        self.scalarFieldil8[:] = np.reshape(cytokines[0], (nx,ny,1), 'F')
        self.scalarFieldil1[:] = np.reshape(cytokines[1], (nx,ny,1), 'F')
        self.scalarFieldil6[:] = np.reshape(cytokines[2], (nx,ny,1), 'F')
        self.scalarFieldil10[:] = np.reshape(cytokines[3], (nx,ny,1), 'F')
        self.scalarFieldtnf[:] = np.reshape(cytokines[4], (nx,ny,1), 'F')
        self.scalarFieldtgf[:] = np.reshape(cytokines[5], (nx,ny,1), 'F')
        
        # Wipe the cells off the field
        cellpresente[:] = 0
        cellpresentndn[:] = 0
        cellpresentna[:] = 0
        cellpresentm1[:] = 0
        cellpresentm2[:] = 0
        
        # Create new lists for cytokine concentrations
        il8_list = []
        il1_list = []
        il6_list = []
        il10_list = []
        tnf_list = []
        tgf_list = []  
        
        # For all the cell types give a position
        for cell in self.cell_list:
            xCOM = int(cell.xCOM)
            yCOM = int(cell.yCOM)
            zCOM = int(cell.zCOM)
            
            # Check if cell is within boundaries
            if xCOM>=500:
                xCOM = 499
            if yCOM>=500:
                yCOM = 499
            # if zCOM>=500:
                # zCOM = 499
                
            # Place different cytokines in the field
            ccil8 = self.scalarFieldil8[xCOM,yCOM,zCOM]
            ccil1 = self.scalarFieldil1[xCOM,yCOM,zCOM]
            ccil6 = self.scalarFieldil6[xCOM,yCOM,zCOM]
            ccil10 = self.scalarFieldil10[xCOM,yCOM,zCOM]
            cctnf = self.scalarFieldtnf[xCOM,yCOM,zCOM]
            cctgf = self.scalarFieldtgf[xCOM,yCOM,zCOM]
            
            # Name the folder and file to store the concentration values
            fileDir=os.path.dirname(os.path.abspath(fullFileName))
            cytoname= fileDir+"/datafiles"+str(mcs)+"concentration.txt"
            
            # Check if the folder exists
            if not os.path.exists(cytoname):
                # Write values to file
                with open(fileDir+"/datafiles"+str(mcs)+"concentration.txt",'w') as cytofile:
                    writer = csv.writer(cytofile)
                    writer.writerow(["mcsteps","xCOM","yCOM","zCOM","il8","il1","il6","il10","tnf","tgf"])
            # Write rows of values per mcs        
            with open(fileDir+"/datafiles"+str(mcs)+"concentration.txt",'a') as cytofile:
                cytowriter = csv.writer(cytofile)
                cytowriter.writerow([str(mcs),str(xCOM),str(yCOM),str(zCOM),str(ccil8),str(ccil1),str(ccil6),str(ccil10),str(cctnf),str(cctgf)])
            # Only add values that are in the tissue area
            if 50<xCOM<450 and 50<yCOM<450:
                il8_list.append(ccil8)
                il1_list.append(ccil1)
                il6_list.append(ccil6)
                il10_list.append(ccil10)
                tnf_list.append(cctnf)
                tgf_list.append(cctgf)
            # If the life of certain cell is bigger than the span, kill it
            if cell.dict["life"] > cell.dict["span"]:
                self.delete_cell(cell)
            # Otherwise   
            else:
                # If the current life of the cell is bigger than its time to grow multiplied by its span, then increase its volume
                if cell.dict["life"]>vv.timeforgrowth*cell.dict["span"]:
                    cell.targetVolume = 2
                    
                # If its a neutrophil, it transitions according to a proba
                if cell.type == 2: # 2 neutrophil-->5 neutrophila
                    proba = sigmoid(ccil8*10**9, vv.sigmoida, vv.sigmoidb)*vv.lnril8 + sigmoid(ccil6*10**11, vv.sigmoida, vv.sigmoidb)*vv.lnril6 + sigmoid(ccil1*10**9, vv.sigmoida, vv.sigmoidb)*vv.lnril1 + sigmoid(cctnf*10**9, vv.sigmoida, vv.sigmoidb)*vv.lnrtnf - sigmoid(ccil10*10**12, vv.sigmoida, vv.sigmoidb)*vv.tnril10
                    if proba > random.random():
                        # Cell type changes
                        cell.type = 5
                        
                # If its a resting monocyte, it transitions according to a proba       
                if cell.type == 7:  # 7 monocyter-->8 macrophage1
                    proba = 0.1 + 0.9*sigmoid(ccil6*10**11, vv.sigmoida, vv.sigmoidb)*vv.lmril6 + sigmoid(cctnf*10**9, vv.sigmoida, vv.sigmoidb)*vv.lmrtnf - sigmoid(ccil10*10**12, vv.sigmoida, vv.sigmoidb)*vv.tmril10 
                    if proba > random.random():
                        # Cell type changes and its characteristics are updated
                        cell.type = 8
                        cell.dict["life"] = 0 
                        cell.dict["span"] = vv.lifespanmr
                        
                # If its a macrophage type 1, it transitions according to a base proba       
                if cell.type == 8:  # 8 macrophage1--> 9 macrophage2 (have base prob)
                    proba = 0.1 + 0.9*sigmoid(ccil10*10**12, vv.sigmoida, vv.sigmoidb)*vv.lm1il10
                    if proba > random.random():
                        # Cell type changes and its characteristics are updated
                        cell.type = 9
                        cell.dict["life"] = 0 
                        cell.dict["span"] = modmm(mcs, vv.km, vv.vmax)    
                
                # If its a fibroblast, it transitions according to a proba
                if cell.type == 4:#4 fibroblast-->10 myofibroblast
                    proba = 0.1*sigmoid(cctgf*10**10, vv.sigmoida, vv.sigmoidb)*vv.lftgf
                    if proba > random.random():
                        # Cell type changes and its characteristics are updated
                        cell.type = 10
                        
                # If its an activated neutrophil, it transitions into an ndn neutrophil according to....        
                if cell.type == 5: # 5 neutrophila-->6 neutrophilndn # needs fixing?
                    if random.randint(0,1000) == 50:
                        # Cell type changes and its characteristics are updated
                        cell.type = 6
                        
                # If its a monocyte, it transitions into a resting monocyte according to base proba
                if cell.type == 3: # 3 monocyte--> 7 monocyter (have base prob)
                    proba = 0.1 + 0.9*sigmoid(ccil6*10**11, vv.sigmoida, vv.sigmoidb)*vv.tranril6
                    if proba > random.random():
                        # Cell type changes and its characteristics are updated
                        cell.type = 7
                        cell.dict["life"] = 0 
                        cell.dict["span"] = vv.lifespanmr
                # Cell count is added per cell type
                ccount[cell.type]+=1
        
        
        # Plot the known count for each cell type at certain mcs per relaxationmcs
        labels = ['Endothelial', 'Neutrophils', 'Monocytes', 'Fibroblast', 'Neutrophil A', 'ND Neutrophil', 'Monocyte R', 'Macrophage I', 'Macrophage II', 'Myofibroblast']
        colors = ["blue", "brown", "cyan", "violet", "red", "pink", "yellow", "orange", "darkblue", "green"]
        [self.plot_win.add_data_point(labels[i], mcs, ccount[i+1]) for i in range(len(labels))]


        
        # Define name of the folder and file to be outputed
        fileDir=os.path.dirname (os.path.abspath(fullFileName))
        namer=fileDir+"/datafiles"+str(mcs)+".png"
        
        # If the folder does not exists, create it
        if not os.path.exists(os.path.dirname(namer)):
            os.makedirs(os.path.dirname(namer))
            
        # Save the plot as a png
        self.plot_win.save_plot_as_png(namer, 1200, 1200)
        
        # Give a name to the file where cell counts are saved
        countname = fileDir+"/cellcount.txt"
        
        # Check if folder/file exists
        if not os.path.exists(countname):
            # Open it
            with open(fileDir+"/cellcount.txt",'w') as file:
                # Write the data
                writer = csv.writer(file)
                writer.writerow(["mcsteps","1","2","3","4","5","6","7","8","9","10"])
                
        # Update the data
        with open(fileDir+"/cellcount.txt", "a") as file:
            writer = csv.writer(file)
            writer.writerow([str(mcs),str(ccount[1]),str(ccount[2]),str(ccount[3]),str(ccount[4]),str(ccount[5]),str(ccount[6]),str(ccount[7]),str(ccount[8]),str(ccount[9]),str(ccount[10])])
        
        # Calculate the means of the cytokine concentrations
        il8_mean = np.mean(il8_list)
        il1_mean = np.mean(il1_list)
        il6_mean = np.mean(il6_list)
        il10_mean = np.mean(il10_list)
        tnf_mean = np.mean(tnf_list)
        tgf_mean = np.mean(tgf_list)
        
        # Calculate the standard deviations of the cytokine concentrations
        il8_std = np.std(il8_list)
        il1_std = np.std(il1_list)
        il6_std = np.std(il6_list)
        il10_std = np.std(il10_list)
        tnf_std = np.std(tnf_list)
        tgf_std = np.std(tgf_list)
        
        # Give a name to the file where mean concentrations are saved
        cytodata = fileDir+"/mean_concentration.txt"
        
        # Check if the folder/file exist
        if not os.path.exists(cytodata):
            with open(fileDir+"/mean_concentration.txt",'w') as cytofile:
                # Write the data on the file and save it
                cytowriter = csv.writer(cytofile)
                cytowriter.writerow(["meanconcen","il8mean","il1mean","il6mean","il10mean","tnfmean","tgfmean","il8std","il1std","il6std","il10std","tnfstd","tgfstd"])
        # Open the file 
        with open(fileDir+"/mean_concentration.txt", "a") as cytofile:
            # Write the data as a string into the file
            cytowriter = csv.writer(cytofile)
            cytowriter.writerow([str(mcs),str(il8_mean),str(il1_mean),str(il6_mean),str(il10_mean),str(tnf_mean),str(tgf_mean),str(il8_std),str(il1_std),str(il6_std),str(il10_std),str(tnf_std),str(tgf_std)])
        
        
        
            
            
    def finish(self):
        """
        Finish Function is called after the last MCS
        """

    def on_stop(self):
        # this gets called each time user stops simulation
        return


        
class celldivisionSteppable(MitosisSteppableBase):
    def __init__(self,frequency=int(vv.relaxationmcs)):
        MitosisSteppableBase.__init__(self,frequency)

    def step(self, mcs):

        cells_to_divide=[]
        for cell in self.cell_list:
            # about cell division!!
            # if cell.volume>=2 and cell.dict["life"] > vv.timeforgrowth*cell.dict["span"] and random.randint(0,int(cell.dict["dividepr"])) == cell.dict["divide"]:
            if cell.volume>=2 and cell.dict["life"] > vv.timeforgrowth*cell.dict["span"] and random.randint(0,1000) == 5000:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divide_cell_along_major_axis(cell)

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume = 1.0                
        self.parent_cell.dict["life"] = 0
        self.clone_parent_2_child()            

        # for more control of what gets copied from parent to child use cloneAttributes function
        # self.clone_attributes(source_cell=self.parent_cell, target_cell=self.child_cell, no_clone_key_dict_list=[attrib1, attrib2]) 
        
        

        

