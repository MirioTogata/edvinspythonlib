# Edvins Python Library
My library for personally coded python functions and classes
<br> Is sorted into my courses right now
<br> The library isn't superrr useful, but it's fun

### Installation
```
pip install edvinspythonlib
```

### Get started
How to use one of my functions or classes in your own code

```Python
# Import one of the modules
from edvinspythonlib import machinelearning as ml

# Call the method you wish to use
model = ml.linearRegression()
model.fit(X,y)
model.residualPlot(X,y)
```

### Documentation
Currently the three working modules are:
* machinelearning
* imageanalysis
* algodata

Later I hope to also add funtionality for:
* mathstat
* computervision
* reinforcementlearning
* activemachinelearning

Testing is currently sparse, but I hope to work on that soon

### machinelearning
Some functions and classes for my course "Machine learning and data mining"
Current implementations:
* function standardizedata
* function PCAe
* function kFoldCV
* function nestedCV
* class linearRegression
* class baselineRegression

### algodata
Some functions and classes for my course "Algorithms and data structures"
Current implementations:
* class Stack
* class Queue
* function SA

### imageanalysis
Some functions and classes for my course "Image analysis"
Current implementations:
* function plotimg
* function todecimal
* function tobin
* function histogram
* function changevectorbasis
* function thinlensequation
* function runlengthencoding
* function chaincode
* function stretchimage
* function stretchfunction
* function gammamapping
* function logmapping
* function bilinear
* function calculateR
* function calculateentropy