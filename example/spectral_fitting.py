import sys
import os
home_dir = os.path.expanduser('~')
sys.path.append(f"{home_dir}/Research/programs/lilpPXF")
import lilpPXF as lilpPXF


import cProfile

def run():
	lilpPXF.run()

	

if __name__ == "__main__":
	# cProfile.run('run()',sort='cumtime')
	run()


