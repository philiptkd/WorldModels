import pstats
import os

profile_dir = "profiles/"
filenames = os.listdir(profile_dir)
for filename in filenames:
    try:
        p = pstats.Stats(profile_dir+filename)
        p.sort_stats('time').print_stats(20)
    except:
        print(profile_dir+filename)
