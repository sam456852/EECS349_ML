from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node)
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  cList = [example['Class'] for example in examples]
  if cList.count(cList[0]) == len(cList):
      return cList[0]
  if (len(examples[0]) == 1):
      return findMajority(cList)
  attribute = attributeToSplit(examples)
  tree = Node()
  tree.label = attribute
  attributeVals = [example[attribute] for example in examples]
  attributeVals = set(attributeVals)
  if '?' in attributeVals:
      attributeVals.remove('?')


  for value in attributeVals:
      tree.children[value] = ID3(splitDataByVal(examples, attribute, value), 0)

  return tree


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  cList = [example['Class'] for example in examples]
  if '?' in cList:
      classifies.remove('?')
  baseAcc = test(node, examples)
  for key in node.children.keys():
      tmp = node.children.pop(key)
      pruned = None
      for res in cList:
          node.children[key] = res
          if test(node, examples) >= baseAcc:
              baseAcc = test(node, examples)
              pruned = res
      if pruned == None:
          node.children[key] = tmp

  for each in node.children.values():
      if isinstance(each, Node):
          prune(each, examples)


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  dataset = dropUnclear(examples)

  total = len(dataset)
  correct = 0.0
  for example in dataset:
      if evaluate(node, example) == example['Class']:
          correct += 1
  return correct / total


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  while isinstance(node, Node):
      #key1 = node.label
      key2 = example[node.label]
      if key2 not in node.children.keys():
          node = node.children.values().pop(0)
      else:
          node = node.children[key2]
      #node = key3
      #node = node.children[example[node.label]]
  return node

def calcEnt(dataSet):
    num = len(dataSet)
    labelCounts = {}
    for data in dataSet:
        label = data["Class"]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    ent = 0.0
    for key in labelCounts:
        prb = float(labelCounts[key])/num
        ent -= prb * math.log(prb,2)
    return ent

def splitDataByVal(dataset, attribute, val):
    subkey = dataset[0].keys()
    subkey.remove(attribute)
    resDataSet = []
    for data in dataset:
        if data[attribute] == val:
            resDataSet.append({key:data[key] for key in subkey})
    return resDataSet

def attributeToSplit(dataset):
    attributes = dataset[0].keys()
    attributes.remove('Class')
    totalData = len(dataset)
    baseEnt = calcEnt(dataset)
    maxInfoGain = 0
    maxFeature = None
    for i in attributes:
        valuesCount = {}
        for data in dataset:
            if data[i] not in valuesCount.keys():
                valuesCount[data[i]] = 0
            valuesCount[data[i]] += 1
        if valuesCount.has_key('?'):
            for value in valuesCount.keys():
                if value != '?':
                    valuesCount[value] += valuesCount[value] * valuesCount['?'] / (totalData - valuesCount['?'])
            del valuesCount['?']

        tmpEnt = 0
        for value in valuesCount.keys():
            tmpDataSet = splitDataByVal(dataset, i, value)
            tmpEnt += calcEnt(tmpDataSet) * len(tmpDataSet) / totalData
        tmpInfoGain = baseEnt - tmpEnt
        if (tmpInfoGain > maxInfoGain):
            maxInfoGain = tmpInfoGain
            maxFeature = i
    return maxFeature

def findMajority(dataset):
    classCount = {}
    for data in dataset:
        if data not in dataset.keys():
            classCount[data] = 0
        classCount[data] += 1
    return max(classCount)

def drawTree(tree):
    if not isinstance(tree, Node):
        return tree
    nextTree = {}
    newTree = {}
    for key in tree.children.keys():
        nextTree[key] = drawTree(tree.children[key])
    newTree[tree.label] = nextTree
    return newTree

def dropUnclear(dataset):
    res = []
    for data in dataset:
        if '?' not in data.values():
            res.append(data)
    return res
