import numpy as np
#import sys
from scipy.spatial.distance import cdist

#reload(sys)
#sys.setdefaultencoding('utf8')

def selectFirstNsegs(sim, anchorVid, targetSegs, N, run, Dict):
	ind_sortd = np.argsort(-sim)
	res_ind = ind_sortd[:N]
	res_sim = np.zeros(N)		
	res_seg = []
	res_targetVids = []
	#
	for i in range(N):
		res_sim[i] = sim[res_ind[i]]
		res_seg.append(targetSegs[res_ind[i]])
		res_targetVids.append(targetSegs[res_ind[i]].strip('\n').split('_')[0])
	#
	result = []
	j = 1
	for i in range(N):
		if res_targetVids[i] != Dict[anchorVid]:
			st = int(res_seg[i].strip('\n').split('_')[1])
			st_m = st/60
			st_s = st%60
			et = int(res_seg[i].strip('\n').split('_')[2])
			et_m = et/60
			et_s = et%60
			txt = '%s Q0 %s %02d.%02d %02d.%02d %d %.12f %s\n'%(anchorVid, res_targetVids[i], st_m, st_s, et_m, et_s, j, res_sim[i], run)
			result.append(txt)
			j = j + 1
	print j-1
	return result

file1 = '../features/anchor_vid_dict.txt'
fr1 = open(file1,'r')
D = eval(fr1.read())
fr1.close()
	
print "Loading similarity data"

simPath = 'sim.npy'


S = np.load(simPath)
#
fr1 = open('../features/anchor1to28Name.txt', 'r')
anchors = fr1.readlines()
fr1.close()
'''
fr = open('/home/robin/LNK/target_files/Traning_Final/Data_Triplet_Bidnn/train/targetEngTxtKFCommonSeg2_fine.txt', 'r')
segsFinal = fr.readlines()
fr.close()
'''
#
targetSegPath = '../features/target_PoolA_names.txt'
fr2 = open(targetSegPath, 'r')
Segs = fr2.readlines()
fr2.close()
#
if S.shape[1] != len(Segs) or S.shape[0] != len(anchors):
	raise ValueError("The number of target segs doesnot match the similarity matrix S (m*n)")
#
'''
#---------------fine---------
print "Adjusting"
S_final = np.zeros((S.shape[0], len(segsFinal)))

for j, one in enumerate(segsFinal):
	ind = Segs.index(one)
	S_final[:, j] = S[:, ind]
#----------------------------
'''
result_all = []
NUM = 1000

for i in range(len(anchors)): 
	anchorname = anchors[i].strip('\n').split()[0]
	print anchorname
	result = selectFirstNsegs(S[i,:], anchorname, Segs, NUM, 'WV', D)
	result_all.extend(result)

# save result
comname = simPath.split('/')[-1][:-4]
resultFile = 'Res%d_%s.txt'%(NUM, comname)
fw = open(resultFile, 'w')
for rows in result_all:
	for clum in rows:
		fw.write(clum)
fw.close()
