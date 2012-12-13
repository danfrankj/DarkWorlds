import optimizesky
import datetime
from haloness import *

TEST_SKIES = range(1, 121)
TRAIN_SKIES = range(1,301)

def create_submission(test=True, kernel=exppow()): 
    if (test):
        SKY_LIST = TEST_SKIES
    else:
        SKY_LIST = TRAIN_SKIES
    timestamp = str(datetime.datetime.now())
    with file('submission' + timestamp.replace(' ', '_').replace(':', '.') + '.csv', 'w') as out:
        for skynum in SKY_LIST:
            print "working on Sky" + str(skynum) + "..."
            dm_x, dm_y, val = optimizesky.predict(skynum, test=test, kernel=kernel, has_width=true)
            # convert from dm_x, dm_y to [x1,y1,x2,y2,...]
            halo_coords = [0.0] * 3 * 2
            for idm in range(dm_x.size):
                halo_coords[idm*2] = dm_x[idm]
                halo_coords[idm*2+1] = dm_y[idm]
                
            print "dm_x = " + str(dm_x) + ", dm_y = " + str(dm_y)
            print halo_coords
            
            sky_id = 'Sky' + str(skynum)
            halo_strs = [str(x) for x in halo_coords]
            out.write(sky_id + ',' + ','.join(halo_strs) + "\n")
            out.flush()

if __name__ == '__main__':
    create_submission()
