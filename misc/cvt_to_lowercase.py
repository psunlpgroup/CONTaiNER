#makeshift
directories = ['data/few-nerd/inter/train.txt', 'data/few-nerd/intra/train.txt']

for file1 in directories:
  final_str = ""
  with open(file1, 'r') as f:
    lines = f.readlines()
    for line in lines:
      cmpnnts = line.split('\t')
      if len(cmpnnts) < 2:
        final_str += line 
        continue
      c1, c2 = cmpnnts
      c1 = c1.lower()
      final_str += (c1 + '\t' + c2 )

  f = open(file1, 'w')
  f.write(final_str)
  f.close()