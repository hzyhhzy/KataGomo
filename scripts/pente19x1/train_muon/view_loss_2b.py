baseDir="../data/train/"
lossItems={"Ip0loss":(2.9,3.05),"Ivloss":(0.4,0.7),"loss":(47,49),"Ipacc1":(0.26,0.35),"gnorm_batch":(0,0),"exgnorm":(0,0),"norm_normal_batch":(0,0),"pslr_batch":(1e-7,1e-1)}#name,ylim,  0 means default
#trainDirs=["b12c240nb1t_adam2_lr2","b6c64d1","b6c64sgd3b"]
trainDirs=["b6c64d2","b12c240nb1t4","b6c64sgd3a","b12c240nb1t_adam2_lr2"]
trainDirs=["b6c64d2","b6c64d3","b6c64d4","b12c240nb1t4"]
trainDirs=["b6c64d4"]
trainDirs=["b2c16rd2","b2c16rd1adam","b2c16rd1sgd","b2c16rd3","b2c16d1adam"]

autoBias=False
biases=[0,0,0]
scales=[1,1,1]
#lossTypes=["train","val"]
lossTypes=["train","val","val_swa0"]
lossTypes=["train"]
outputFile="../loss_2b.png"

logPlot=True
logPlotXmin=1e7
#logPlotXmin=2.5e9
#logPlotXmax=3e9
logPlotXmax=None
smooth_window=10
smooth_window_val=0


lossKeys=list(lossItems.keys())
nKeys=len(lossKeys)



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
            continue
        if("nsamp_train" in j):
            nsamp=j["nsamp_train"]
        else:
            nsamp = j["nsamp"]

        d["nsamp"].append(nsamp)
        for key in lossKeys:
            if(key not in j):
                continue
            d[key].append(j[key])
    return d



#os.makedirs(outputDir,exist_ok=True)


fig=plt.figure(figsize=(12,12*nKeys),dpi=100)
plt.subplots_adjust(hspace=0.5)
for i in range(nKeys):
    key=lossKeys[i]
    ax=plt.subplot(nKeys,1,i+1)

    plotLim=lossItems[key]
    ax.set_xlabel("nsamp")
    ax.set_ylabel(key)
    ax.set_title(key)

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
        
maxX=0
isSingleDir = len(trainDirs)==1
if(isSingleDir):
    fig.suptitle(trainDirs[0])
for trainDirId in range(len(trainDirs)):
    trainDir=trainDirs[trainDirId]
    b=0
    s=1
    if biases is not None and len(biases)>trainDirId:
        b=biases[trainDirId]
    if scales is not None and len(scales)>trainDirId:
        s=scales[trainDirId]
    for lossType in lossTypes:
        jsonPath=os.path.join(baseDir,trainDir,"metrics_"+lossType+".json")
        if(not os.path.exists(jsonPath)):
            print("warning:",jsonPath,"not exists")
            continue
        jsonData=readJsonFile(jsonPath,lossKeys)

        for i in range(nKeys):
            key = lossKeys[i]
            if(key not in jsonData or len(jsonData[key])==0):
                continue
            ax = plt.subplot(nKeys, 1, i + 1)
            plotLabel=lossType if isSingleDir else trainDir+"."+lossType
            xdata=jsonData["nsamp"]
            xdata=[s*x+b for x in xdata]
            #if(trainDir=="b12c64n2n"):
            #    xdata=[x+7e9 for x in xdata]
            if(trainDirId==0):
                maxX=max(xdata)
            if(autoBias):
                b1=maxX-max(xdata)
                xdata=[x+b1 for x in xdata]
            from scipy.signal import savgol_filter
            ydata=jsonData[key]
            smooth_window_this=smooth_window_val
            if(lossType=="train"):
                smooth_window_this = smooth_window
            if smooth_window_this>0:
                ydata = savgol_filter(ydata, window_length=smooth_window_this, polyorder=1)  # Adjust window_length and polyorder as needed
    
            ax.plot(xdata, ydata, label=plotLabel)
            #ax.scatter(xdata, ydata, label=plotLabel, s=1, marker='o')
            ax.legend(loc="upper right")
            if logPlot:
                plt.xscale('log')
                ax.set_xlim(logPlotXmin,logPlotXmax)
            if(key=="pslr_batch"):
                plt.yscale('log')


plt.tight_layout()
plt.savefig(outputFile)