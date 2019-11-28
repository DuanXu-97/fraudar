import argparse
import time
start_time = time.time()
import fraudar
from density_metrics import *
import load_data as ld
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help="Path of data file")
parser.add_argument('--output_path', type=str, required=True, help="Path of Output files")
parser.add_argument('--density_metric', type=str, default='LogWeightedAveDegree', help="Density metric to be used")
parser.add_argument('--node_sus_path', type=str, default=None, help="Path of node suspiciousness file")

args = parser.parse_args()

M = ld.load_data(args.data_path)
print("matrix shape: ", M.shape)

if args.density_metric == 'LogWeightedAveDegree':
    dm = LogWeightedAveDegree(matrix=M, c=5)
elif args.density_metric == 'SqrtWeightedAveDegree':
    dm = SqrtWeightedAveDegree(matrix=M, c=5)
elif args.density_metric == 'AveDegree':
    dm = AveDegree(matrix=M)
else:
    raise Exception("Error Density Metric.")

matrix = dm.get_matrix()
lwRes = fraudar.run_fraudar(matrix, dm, numToDetect=10)

print("result: ", lwRes)
np.savetxt("%s.rows" % (args.output_path, ), np.array(list(lwRes[0][0])), fmt='%d')
np.savetxt("%s.cols" % (args.output_path, ), np.array(list(lwRes[0][1])), fmt='%d')