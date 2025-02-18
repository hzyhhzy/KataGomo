baseDir="../data/train/"
lossItems={"p0loss":(1.4,2.2),"vloss":(0.5,0.9),"loss":(44.7,45.2),"gnorm_batch":(0,2e3),"exgnorm":(0,3e2),"norm_normal_batch":(0,0)} #name,ylim,  0 means default
trainDirs=["b10c384n"];
lossTypes=["train","val"]
outputFile="../loss.png"



lossKeys=list(lossItems.keys())
nKeys=len(lossKeys)

smooth_window=0


import json
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def readJsonFile(path,lossKeys):
    d={}
    d["nsamp"]=[]
    for key in lossKeys:
        d[key]=[]

    f=open(path,"r")
    filelines=f.readlines()
    for line in filelines:
        if(len(line)<5):
            continue #bad line
        j=json.loads(line)
        if("p0loss" not in j):
            continue #bad line
        if("nsamp_train" in j):
            nsamp=j["nsamp_train"]
        else:
            nsamp = j["nsamp"]

        d["nsamp"].append(nsamp)
        for key in lossKeys:
            if key in j:
                d[key].append(j[key])
    return d



#os.makedirs(outputDir,exist_ok=True)


fig=plt.figure(figsize=(12,8*nKeys),dpi=100)
plt.subplots_adjust(hspace=0.5)
for i in range(nKeys):
    key=lossKeys[i]
    ax=plt.subplot(nKeys,1,i+1)
    plotLim=lossItems[key]
    ax.set_xlabel("nsamp")
    ax.set_ylabel(key)
    ax.set_title(key)
    #ax.set_xlim(5.63e9,5.64e9)
    if(plotLim[0]!=0 or plotLim[1]!=0):
        ax.set_ylim(plotLim[0],plotLim[1])
        y_major_locator=None
        y_minor_locator=None
        if(plotLim[1]-plotLim[0]>5):
            pass
        elif(plotLim[1]-plotLim[0]>2):
            y_major_locator=MultipleLocator(0.5)
            y_minor_locator=MultipleLocator(0.1)
        elif(plotLim[1]-plotLim[0]>0.5):
            y_major_locator=MultipleLocator(0.1)
            y_minor_locator=MultipleLocator(0.02)
        elif(plotLim[1]-plotLim[0]>0.25):
            y_major_locator=MultipleLocator(0.05)
            y_minor_locator=MultipleLocator(0.01)
        elif(plotLim[1]-plotLim[0]>0.10):
            y_major_locator=MultipleLocator(0.02)
            y_minor_locator=MultipleLocator(0.01)
        elif(plotLim[1]-plotLim[0]>0.02):
            y_major_locator=MultipleLocator(0.01)
            y_minor_locator=MultipleLocator(0.002)
        elif(plotLim[1]-plotLim[0]>0.01):
            y_major_locator=MultipleLocator(0.005)
            y_minor_locator=MultipleLocator(0.001)
        if(y_major_locator is not None):
            ax.yaxis.set_major_locator(y_major_locator)
            ax.yaxis.set_minor_locator(y_minor_locator)

isSingleDir = len(trainDirs)==1
if(isSingleDir):
    fig.suptitle(trainDirs[0])
for trainDir in trainDirs:
    for lossType in lossTypes:
        jsonPath=os.path.join(baseDir,trainDir,"metrics_"+lossType+".json")
        jsonData=readJsonFile(jsonPath,lossKeys)

        for i in range(nKeys):
            key = lossKeys[i]
            if(key not in jsonData):
                continue
            if(len(jsonData[key])==0):
                continue
            ax = plt.subplot(nKeys, 1, i + 1)
            plotLabel=lossType if isSingleDir else trainDir+"."+lossType

            xdata=jsonData["nsamp"]

            from scipy.signal import savgol_filter
            ydata=jsonData[key]
            if smooth_window>0:
                ydata = savgol_filter(ydata, window_length=smooth_window, polyorder=2)  # Adjust window_length and polyorder as needed
    
            #ax.plot(xdata, ydata,'.', label=plotLabel)
            ax.plot(xdata, ydata, label=plotLabel)
            
            ax.legend(loc="upper right")


plt.savefig(outputFile)