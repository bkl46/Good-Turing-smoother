from collections import Counter, defaultdict

mine = [1,2,2,2,2,2,1,12,21,4,1,1,4,2,4,5,31,25,1,4,6,7,8,5,6,6,7,5,4,1,1,103,4,5,6,8,8,9,2,56]

counts = Counter(mine)

print(counts)

pr = {}

for key in counts.keys():
    pr[key] = counts[key]/len(mine)

print(pr)
print(sum(pr.values()))

Phi = defaultdict(int)

for item, c in counts.items():
    Phi[c] +=1
Phi = dict(sorted(Phi.items()))

print(Phi)
gt_mass ={}
gt_mass[0] = Phi.get(2,0)/len(mine)

print(gt_mass)