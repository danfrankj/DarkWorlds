# HAD TO
import optimizesky
import datetime

TEST_SKIES = range(1, 121)


def create_submission():
    with file('submission' + str(datetime.datetime.now()) + '.csv', 'w') as out:
        for skynum in TEST_SKIES:
            halo_coords = optimizesky.predict(skynum, test=True)
            sky_id = 'Sky' + str(skynum)
            halo_strs = [str(x) for x in halo_coords]
            out.write(sky_id + ',' + ','.join(halo_strs) + "\n")
            out.flush()

if __name__ == '__main__':
    create_submission()
