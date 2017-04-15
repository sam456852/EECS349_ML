from node import Node
import math, parse, random
import numpy as np
import pylab as pl


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

  # decide which attribute to split
  attribute = attributeToSplit(examples)
  tree = Node()
  tree.label = attribute
  attributeVals = [example[attribute] for example in examples]
  attributeVals = set(attributeVals)
  if '?' in attributeVals:
      attributeVals.remove('?')


  for value in attributeVals:
      tree.children[value] = ID3(splitDataByVal(examples, attribute, value), default)

  return tree


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''

  # depth-first traverse
  for key in node.children.keys():
      if isinstance(node.children[key], Node) and len(splitDataByVal(examples, node.label, key)) > 0:
          prune(node.children[key], splitDataByVal(examples, node.label, key))

  cList = [example['Class'] for example in examples]
  if '?' in cList:
      cList.remove('?')

  baseAcc = test(node, examples)
  newKey = None
  maxPruned = None
  maxAcc = 0
  for key in node.children.keys():
      tmp = node.children.pop(key)
      pruned = None
      for res in cList:
          node.children[key] = res
          #if test(node, examples) > baseAcc:
          if baseAcc < test(node, examples) and maxAcc < test(node, examples):
              #baseAcc = test(node, examples)
              maxAcc = test(node, examples)
              maxPruned = res
              newKey = key
      node.children[key] = tmp
  if maxPruned != None and newKey != None:
      node.children[newKey] = maxPruned



def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  dataset = dropUnclear(examples)

  total = len(dataset)
  if total == 0 :
      return 0.0
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

      key2 = example[node.label]
      if key2 not in node.children.keys():
          node = node.children.values().pop(0)
      else:
          node = node.children[key2]

  return node

def calcEnt(dataSet):
    '''
    calculate the Entropy for given dataset
    '''
    num = len(dataSet)
    labelCounts = {}
    for data in dataSet:
        label = data["Class"]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    ent = 0.0
    for key in labelCounts.keys():
        prb = float(labelCounts[key])/num
        ent -= prb * math.log(prb,2)
    return ent

def splitDataByVal(dataset, attribute, val):
    '''
    split given dataset by given attribute and value
    '''
    subkey = dataset[0].keys()
    subkey.remove(attribute)
    resDataSet = []
    for data in dataset:
        if data[attribute] == val:
            resDataSet.append({key:data[key] for key in subkey})
    return resDataSet

def attributeToSplit(dataset):
    '''
    select most significant attribute for given dataset
    '''
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
    '''
    find the most commom label for given dataset
    '''
    classCount = {}
    for data in dataset:
        if data not in classCount.keys():
            classCount[data] = 0
        classCount[data] += 1
    return max(classCount)

def drawTree(tree):
    '''
    print out the given tree, used for test
    '''
    if not isinstance(tree, Node):
        return tree
    nextTree = {}
    newTree = {}
    for key in tree.children.keys():
        nextTree[key] = drawTree(tree.children[key])
    newTree[tree.label] = nextTree
    return newTree

def dropUnclear(dataset):
    '''
    drop value '?' in given dataset
    '''
    res = []
    for data in dataset:
        if '?' not in data.values():
            res.append(data)
    return res

def drawTest(inFile):
      '''
      plot the accuracy
      '''
      prunedSum = []
      withoutPrunedSum = []

      trainLenSum = []
      data = parse.parse(inFile)
      dataLen = len(data)

      for trainLen in range(10, 301, 10):
          print trainLen
          withPruning = []
          withoutPruning = []
          trainLenSum.append(trainLen)

          for i in range(100):
            random.shuffle(data)
            train = data[:trainLen]
            valid = data[trainLen:(dataLen + trainLen)/2]
            testP = data[(dataLen + trainLen)/2:]

            tree = ID3(train, 'democrat')

            prune(tree, valid)

            acc = test(tree, testP)

            withPruning.append(acc)
            tree = ID3(train+valid, 'democrat')
            acc = test(tree, testP)

            withoutPruning.append(acc)

          prunedSum.append(sum(withPruning)/len(withPruning))
          withoutPrunedSum.append(sum(withoutPruning)/len(withoutPruning))

      pl.plot(trainLenSum, withoutPrunedSum, 'g', label=u'without pruning')
      pl.plot(trainLenSum, prunedSum, 'r', label=u'pruning')
      pl.legend()
      pl.title("Pruned vs Not Pruned")
      pl.xlabel("Training data size")
      pl.ylabel("Accuracy")

      pl.xlim(0, 300)


      pl.show()


      improved = []

      for i in range(len(prunedSum)):
          improved.append(float(prunedSum[i]) - withoutPrunedSum[i])

      pl.title("Advantage of Pruning")
      pl.xlabel("Training data size")
      pl.ylabel("Accuracy")
      pl.plot(trainLenSum, improved, 'b')
      pl.show()
