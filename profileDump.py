import pstats
FOUT = open('profile.txt', 'w')
stats = pstats.Stats('profile.bin', stream=FOUT)
stats.sort_stats('cumtime')
stats.print_stats()