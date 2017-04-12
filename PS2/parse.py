import csv
import ID3

def parse(filename):
  '''
  takes a filename and returns attribute information and all the data in array of dictionaries
  '''
  # initialize variables

  out = []
  csvfile = open(filename,'rb')
  fileToRead = csv.reader(csvfile)

  headers = fileToRead.next()

  # iterate through rows of actual data
  for row in fileToRead:
    out.append(dict(zip(headers, row)))

  return out
#print ID3.calcEnt(parse("/Users/sam/Documents/NU/349 MACHINE LEARNING/PS2/house_votes_84.data"))
##print ID3.attributeToSplit(parse("/Users/sam/Documents/NU/349 MACHINE LEARNING/PS2/house_votes_84.data"))
