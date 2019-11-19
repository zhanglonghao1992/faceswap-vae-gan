import os
import numpy as np

for p,d,f in os.walk('aligned_faces/'):
    if len(d) > 0:
        id_list = sorted(d)

for i in range(15):
    id = id_list[i]
    #print(id)
    for p,d,f in os.walk('aligned_faces/'+id):
        for im in f:
            #print(im)
            im_path = 'aligned_faces/' + id + '/' + im
            l = np.zeros(15)
            l[i] = 1
            label = str(l)
            line = im_path + '\t' + id + '\t' + label + '\n'

            with open('face.txt', 'a') as fa:
                fa.write(line)
                fa.close()





