# HAD TO
import optimizesky
import datetime
from haloness import *

SKY_LIST = range(1, 120)

def create_submission(test=True):
    timestamp = str(datetime.datetime.now())
    with file('submission' + timestamp.replace(' ', '_').replace(':', '.') + '.csv', 'w') as out:
        for skynum in SKY_LIST:
            print skynum
            halo_coords, val = optimizesky.predict(skynum, test=test, kernel=gaussian(750.))
            sky_id = 'Sky' + str(skynum)
            halo_strs = [str(x) for x in halo_coords]
            out.write(sky_id + ',' + ','.join(halo_strs) + "\n")
            out.flush()

if __name__ == '__main__':
    create_submission()
