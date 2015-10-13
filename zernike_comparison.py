from pylab import *
from os import listdir, getcwd
from os.path import isfile, join
from sklearn.decomposition import FastICA


### Useful for parsing through data files
mypath = getcwd() + '/VS212A_2015_Aberration_Data/ZER/' # choose path for .zer files
onlyfiles = [ f for f in listdir(mypath) if (isfile(join(mypath,f)) and '.zer' in f)] 

mymtfpath = getcwd() + '/WavefrontAnalysisPrograms/images/' # choose path for only MTFs for high order abs
onlymtffiles = [ f for f in listdir(mymtfpath) if (isfile(join(mymtfpath,f)) and '.txt' in f)]

mydefspath = getcwd() + '/VS212A_2015_Aberration_Data/ZER/5/defocus/' # choose this path for defocus/rms/strehl data
onlydefsfiles = [ f for f in listdir(mydefspath) if (isfile(join(mydefspath,f)) and '.txt' in f)]

mydefmtfpath = getcwd() + '/WavefrontAnalysisPrograms/images/defocus_mtf/' #choose path for MTF files across defocus
onlydefmtffiles = [ f for f in listdir(mydefmtfpath) if (isfile(join(mydefmtfpath,f)) and '.txt' in f)]

mystrehlpath = getcwd() + '/WavefrontAnalysisPrograms/images/strehl/' # choose path for MTF files across pupil
onlystrehlfiles =  [ f for f in listdir(mystrehlpath) if (isfile(join(mystrehlpath,f)) and '.txt' in f)]

zern_by_size = {'3':[],
                '4':[],
                '5':[],
                '6':[],
                '7':[],
                '8':[]}

name_by_size = {'3':[],
                '4':[],
                '5':[],
                '6':[],
                '7':[],
                '8':[]}

for fname in onlyfiles:
    # parse through .zer file meta-info and save
    fname_split = fname.split('_')
    pup_size = str(int(fname_split[-1][0:3])/100)
    name = fname_split[0]
    name_by_size[pup_size].append(name)

    
    # save all the zernicke coefficients
    reached_data = False
    f = open(mypath+fname)
    zern_coeffs = []
    for line in f:
        if reached_data:
            datum = line.split()[2]
            zern_coeffs.append(datum)
        elif 'Data' in line:
            reached_data = True

    # group the zernicke coffecients by pupil size, and tag each subject name to each saved loc
    if zern_by_size[pup_size] == []:
        new_arr = ones((1,65))
        new_arr[0,:] = array(zern_coeffs[1:]) #ignore the first term
        zern_by_size[pup_size] = new_arr
    else:
        old_arr = zern_by_size[pup_size]
        new_arr = ones((old_arr.shape[0]+1,old_arr.shape[1]))
        new_arr[0:old_arr.shape[0],:] = old_arr
        new_arr[old_arr.shape[0],:] = array(zern_coeffs[1:]) #ignore the first term
        zern_by_size[pup_size] = new_arr

# parses MTF files for high order abs
mtfs = []
mtf_names = []
for fname in onlymtffiles:
    mtf_names.append(fname.split('_')[0])
    f = open(mymtfpath+fname)
    mtf_f = []
    for line in f:
        mtf_line = []
        mtf_line.append(float(line.split()[0]))
        mtf_line.append(float(line.split()[1]))
        mtf_f.append(mtf_line)
    mtfs.append(array(mtf_f))

# parses defocus/rms/strehl data
defs = []
def_names = []
for fname in onlydefsfiles:
    def_names.append(fname.split('_')[0])
    f = open(mydefspath+fname)
    def_f = []
    for line in f:
        def_line = []
        def_line.append(float(line.split()[0]))
        def_line.append(float(line.split()[1]))
        def_line.append(float(line.split()[2]))
        def_f.append(def_line)
    defs.append(array(def_f))


# parses MTF files across defocuses
def_vals = ['-1','-0.75','-0.5','-0.25','0','0.25','0.5','0.75','1']
dm_names = ['Linyue', 'Kat', 'Jazzi']

def_mtfs = {}
for name in dm_names:
    arr = []
    for ind in range(len(def_vals)):
        arr.append([])
    def_mtfs[name] = arr

def_mtf_names = []
for fname in onlydefmtffiles:
    def_val = fname.split('_')[-2]
    f = open(mydefmtfpath+fname)
    mtf_f = []
    for line in f:
        mtf_line = []
        mtf_line.append(float(line.split()[0]))
        mtf_line.append(float(line.split()[1]))
        mtf_f.append(mtf_line)
    def_mtfs[fname.split('_')[0]][def_vals.index(def_val)].append(array(mtf_f))

# parses MTF files across pupil sizes
mtf_strehl = []
pup_sizes = []
for fname in onlystrehlfiles:
    pup_size = fname.split('_')[-2]
    pup_sizes.append(pup_size)
    f = open(mystrehlpath+fname)
    mtf_s = []
    for line in f:
        mtf_line = []
        mtf_line.append(float(line.split()[0]))
        mtf_line.append(float(line.split()[1]))
        mtf_s.append(mtf_line)
    mtf_strehl.append(array(mtf_s))

### Useful for plotting/calculating things related to Zernike coefficients
def zscore_real(arr):
    zarr = arr.copy()
    zarr -= mean(zarr,axis=0)
    zarr /= std(zarr,axis=0)
    return zarr

all_keys = zern_by_size.keys()
all_zs_len = 0
for key in all_keys:
    all_zs_len += zern_by_size[key].shape[0]

all_zs = zeros((all_zs_len,65))
all_ind = 0
for key in all_keys:
    curr_shape = zern_by_size[key].shape[0]
    all_zs[all_ind:all_ind+curr_shape,:] = zern_by_size[key]
    all_ind += curr_shape

z3 = all_zs
z3_zscored = zscore_real(z3)
z3_zscored[isnan(z3_zscored)] = 0

ica = FastICA(max_iter=10000)
ica2 = FastICA(max_iter=10000)

ica_z3 = ica.fit_transform(z3.T).T
ica_z3_zscored = ica2.fit_transform(z3_zscored.T).T

f = figure()
f.add_subplot(221)
imshow(z3)
f.add_subplot(222)
imshow(ica_z3)
f.add_subplot(223)
imshow(z3_zscored)
f.add_subplot(224)
imshow(ica_z3_zscored)
show()

### Useful for plotting/calculating things related to Zernike coefficients of 5mm pupil size directly
z5 = zern_by_size['5']
name_set = set(name_by_size['5'])
name_inds = []
for name in name_set:
    name_inds.append(name_by_size['5'].index(name)) #takes only the flat 5mm pupil size for each person
name_inds = sort(name_inds)
z5 = z5[name_inds,:]

ica = FastICA(max_iter=10000)

ica_z5 = ica.fit_transform(z5.T).T

r = range(0,65)
m1 = mean(z5,axis=0)
m2 = mean(ica_z5,axis=0)
s1 = std(z5,axis=0)
s2 = std(ica_z5,axis=0)

title('Average for each Zernike Coeff Across Subjects')
xlabel('Coeff'); ylabel('Average')
plot(r,m1,'g-'); show()
title('Average for each Zernike Coeff Across Subjects with Independent Coeffs')
xlabel('Coeff'); ylabel('Average')
plot(r,m2,'r-'); show()

title('Std Dev for each Zernike Coeff Across Subjects')
xlabel('Coeff'); ylabel('Std Dev')
plot(r,s1,'g-'); show()
title('Std Dev for each Zernike Coeff Across Subjects with Independent Coeffs')
xlabel('Coeff'); ylabel('Std Dev')
plot(r,s2,'r-'); show()

rms = []
rms_s = []
rms2 = []
rms_s2 = []
orders = [[0,1],
          [2,3,4],
          [5,6,7,8],
          [9,10,11,12,13],
          [14,15,16,17,18,19],
          [20,21,22,23,24,25,26],
          [27,28,29,30,31,32,33,34],
          [35,36,37,38,39,40,41,42,43],
          [44,45,46,47,48,49,50,51,52,53],
          [54,55,56,57,58,59,60,61,62,63,64]]

for order in orders:
    rms_o = []
    rms_o2 = []
    for i in range(z5.shape[0]):
        rms_o.append(sum(z5[i,order]**2)**.5)
        rms_o2.append(sum(ica_z5[i,order]**2)**.5)

    rms.append(mean(rms_o))
    rms_s.append(std(rms_o)/(len(rms_o)**.5))

    rms2.append(mean(rms_o2))
    rms_s2.append(std(rms_o2)/(len(rms_o2)**.5))

r = range(len(rms))
title('Average Zernike Coeff RMS per Order Across Subjects')
xlabel('Coeff Group'); ylabel('Average')
bar(r, rms, color='g', yerr=rms_s, ecolor='r'); show()
title('Average Zernike Coeff RMS per Order Across Subjects with Independent Coeffs')
xlabel('Coeff Group'); ylabel('Average')
bar(r, rms2, color='g', yerr=rms_s2, ecolor='r'); show()


subj_rms = [0.177652, 0.354648, 0.290274, 0.248618, 0.250094, 0.364105, 0.338985, 0.215131, 0.545866, 0.409116, 0.315491, 0.268575, 0.369134, 0.227085, 0.175701, 0.428314]
subj_strehl = [0.139758, 0.0415934, 0.0445219, 0.0837672, 0.0531977, 0.0531316, 0.0535383, 0.0606863, 0.0449174, 0.0263712, 0.0607622, 0.0366341, 0.0367964, 0.0607025, 0.0921136, 0.0205751]
names = name_by_size['5']

title('High Order Zernike Coeff RMS per Subject')
table(cellText=subj_rms, colLabels=names)

##### Useful for calculating/plotting things related to the MTF/rms files from MATLAB
title('Radial Average MTFs for High Order Aberrations')
xlabel('Spatial Frequency (c/deg)'); ylabel('Contrast')
for i in range(len(mtfs)):
    mtf = mtfs[i]
    plot(mtf[:,0], mtf[:,1], label=mtf_names[i])
    print str(i) + ': ' + str(mtf[10,:])
legend()
show()


title('RMS and Strehl Ratio Over Range of Defocus')
xlabel('Defocus'); ylabel('RMS/Strehl')
for i in range(len(defs)):
    defi = defs[i]
    plot(defi[:,0], defi[:,1], label=def_names[i]+' RMS')
    plot(defi[:,0], defi[:,2], label=def_names[i]+' Strehl')
legend()
show()

for fname in def_mtfs.keys():
    print fname
    arr = def_mtfs[fname]
    title('Radial Average MTFs for High Order Aberrations Across Defocuses for ' + fname)
    xlabel('Spatial Frequency (c/deg)'); ylabel('Contrast')
    for i in range(len(def_mtfs[fname])):
        def_mtf = def_mtfs[fname][i][0]
        print str(def_vals[i]) + ': ' + str(def_mtf[13])#str(trapz(y=def_mtf[:,1],x=def_mtf[:,0]))
        plot(def_mtf[:,0], def_mtf[:,1], label=def_vals[i])
    legend()
    show()

pups = [6, 5, 4, 3, 2]
strehls = [0.0686716, 0.0680597, 0.0679807, 0.070351, 0.0698435]
title('Strehl Ratios Across Pupil Sizes for Sylvain')
xlabel('Pupil Size'); ylabel('Strehl Ratio')
plot(pups,strehls,'g-'); show()

title('Radial Average MTFs for High Order Aberrations at Ideal Defocus Across Pupil Sizes  for Sylvain')
xlabel('Spatial Frequency (c/deg)'); ylabel('Contrast')
for ind in range(len(mtf_strehl)):
    arr = mtf_strehl[ind]
    plot(arr[:,0], arr[:,1], label=pup_sizes[ind])
legend()
show()

### Last calculation for the last question about Thibos' equation
tar = [.44,.525,.63]
for lam in tar:
    K = 1.68524 - (.63346/(lam - .2141))
    print 1/K
