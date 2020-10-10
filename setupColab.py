from broccole.CocoDataset import CocoDataset
from broccole.CocoDatasetBuilder import CocoDatasetBuilder

from broccole.logUtils import init_logging
init_logging('setupColab.log')

humanDataset = CocoDatasetBuilder('../drive/My Drive/broccole/data/annotations/instances_val2017.json', '../val2017').addClasses([1]).build()
CocoDataset.save(humanDataset, '../drive/My Drive/broccole/data/human')

nonHumanDataset = CocoDatasetBuilder('../drive/My Drive/broccole/data/annotations/instances_val2017.json', '../val2017').selectAll().filterNonClasses([1]).build(shuffle=True)
CocoDataset.save(nonHumanDataset, '../drive/My Drive/broccole/data/nonHuman')

valHumanDataset = CocoDatasetBuilder('../drive/My Drive/broccole/data/annotations/instances_val2017.json', '../val2017').addClasses([1]).build()
CocoDataset.save(valHumanDataset, '../drive/My Drive/broccole/data/valHuman')

valNonHumanDataset = CocoDatasetBuilder('../drive/My Drive/broccole/data/annotations/instances_val2017.json', '../val2017').selectAll().filterNonClasses([1]).build(shuffle=True)
CocoDataset.save(valNonHumanDataset, '../drive/My Drive/broccole/data/valNonHuman')