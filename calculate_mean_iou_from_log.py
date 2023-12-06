import sys
file_name = sys.argv[1]
with open(file_name) as f:
  lines = f.read().split('\n')[:-1]

ious = [float(line.split(':')[1][1:]) for line in lines]
print(sum(ious)/len(ious))
