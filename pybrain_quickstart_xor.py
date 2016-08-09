from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer

net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer, bias=True)
net.activate([2, 1])
ds = SupervisedDataSet(2, 1)

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds)

x = trainer.trainUntilConvergence()

import json
print json.dumps(x, indent=4)
