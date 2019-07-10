import matplotlib.pyplot as plt
import numpy
import scipy.stats
from time import time
sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]


filter_index = []
def filter_traces(leakage,NB_samples):
	for i in range (NB_samples):
		if not numpy.unique(leakage[:,i]).size == 1:
			filter_index.append(i)

def main ():
	start_tracing_time = time()
	DIR = "../traces_masked_WB/"
	FileName_input = DIR + "plaintexts.bin"
	NB_traces = 500
	NB_samples = 17952
	input = numpy.fromfile(FileName_input, dtype=numpy.uint8)
	input = input.reshape((NB_traces, 16))
	
	NB_traces_attack = 500
	FileName_leakage = DIR + "500_computation_traces.bin"
	leakage = numpy.fromfile(FileName_leakage, dtype=numpy.uint8)
	leakage = leakage.reshape((NB_traces, NB_samples))
	NB_samples = leakage.shape[1]

	target_sbox = 0
	filter_traces(leakage, NB_samples)
	filtered_leakage = leakage[:NB_traces_attack,filter_index]
	

		
	Good_key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c]
	NB_keys = 256
	recovered_key= 16*[0]
	for target_sbox in range (0, 1):
		weight = numpy.zeros((NB_keys, NB_traces_attack))
		for i in range (NB_traces_attack):
			for key in range (NB_keys):
				weight [key][i] = sbox[input[i][target_sbox]^key]&0xF
		
		index_mask_start = 787
		index_mask_end   = 789
		
		index_data_start = 1190
		index_data_end   = 1200
		NB_samples = (index_data_end - index_data_start ) * (index_mask_end - index_mask_start)
		cond_entropy = numpy.zeros((NB_keys, NB_samples))
		count = 0
		for index_mask  in range (index_mask_start, index_mask_end):
			print ("processing sample number : ", index_mask)
			for index_data in range (index_data_start, index_data_end):
				for key in range (NB_keys):
					for index in range (16):
						tab = numpy.where(weight [key] ==index)
						tmp1 = filtered_leakage[tab,index_data][0]
						tmp2 = filtered_leakage[tab,index_mask][0]
						comb_leakage = numpy.bitwise_xor(tmp1,tmp2)
						hist = numpy.histogram(comb_leakage, bins=16, density=True)[0]
						cond_entropy[key][count]+= (1/16)*scipy.stats.entropy(hist, base=2)
				count+=1
				
		recovered_key[target_sbox] = numpy.where(cond_entropy ==numpy.amin(cond_entropy))[0][0]
		print ("best key for Sbox number : ", target_sbox, " is: ", recovered_key[target_sbox])
		plt.figure()
		for i in range (NB_keys):
			plt.plot(cond_entropy[i], color = 'blue')
		plt.plot(cond_entropy[Good_key[target_sbox]], color = 'red')
		
	print("Complete recovered key : {}".format(bytes(recovered_key).hex()))
	print ("time: ", time() - start_tracing_time)
	plt.show()

if __name__ == "__main__":
    main()
