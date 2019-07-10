from time import time
import struct
import numpy

from bucketing.utils.aes_utils import reverse_key_schedule
from bucketing.utils.viewer import plot_data
from bucketing.utils.aes_utils import SBX, SBX_INV


class Bucketing():

	def __init__(self,
				traces_root_path,
				start_s_box=0,
				end_s_box=16,
				nb_traces=16,
				plot=False,
				decrypt=False,
				verbose=False,
				):
		self.verbose = verbose
		self.start_s_box = start_s_box
		self.end_s_box = end_s_box
		self.traces_root_path = traces_root_path
		self.nb_traces_per_set = nb_traces
		self.decrypt = decrypt
		self.start_guess = 0
		self.end_guess = 256
		self.nb_samples = 0
	
		self.recovered_key = 16*[0]
		self.master_key = None
		self.plot = plot
		self.inputs = []
		self.traces = []
		self.regrouped_traces = []
		self.filter_index = [[] for i in range(16)]
		self.__pre_computation()

	
	def __pre_computation(self):
		sbx = SBX_INV if self.decrypt else SBX
		for s in range(self.start_s_box, self.end_s_box):
			s_traces = []
			s_inputs = []
			path = self.traces_root_path + "/sbx_{}/".format(s)
			for x in range(256):
				s_inputs.append([x if i == s else 0 for i in range(16)])
				s_traces.append(path + "trace_{}".format(x))
			self.inputs.append(s_inputs)
			self.traces.append(s_traces)
			self.nb_samples = len(self.__read_trace(self.traces[0][0]))
			self.filter_traces(s)
			print (self.filter_index[s])
			buckets = []
			for g in range(256):
				sub_bucket = []
				for d in [0x0, 0xf]:
					sub_sub_bucket = []
					for i, p in enumerate(s_inputs):
						if sbx[p[s] ^ g] & 0x0f == d:
						#if sbx[p[s] ^ g] >>4 == d:
							sub_sub_bucket.append(s_traces[i])
					sub_bucket.append(sub_sub_bucket)
				buckets.append(sub_bucket)

			self.regrouped_traces.append(buckets)
	
	@staticmethod
	def __read_trace(file_name):
		f = open(file_name, "rb")
		trace_data = []
		while True:
			e = f.read(1)
			if not e:
				break
			trace_data.append(struct.unpack("B", e)[0])
		f.close()
		return trace_data

	@staticmethod
	def __get_filtered_trace(file_name, filter_index):
		f = open(file_name, "rb")
		trace_data = []
		for i in filter_index:
			f.seek(i)
			trace_data.append(struct.unpack("B", f.read(1))[0])
		f.close()
		return trace_data
	
	def get_filtered_ip0_ip1(self, filter_index, current_s_box, g):
		filtered_ip0, filtered_ip1 = [], []
		for i in range(self.nb_traces_per_set):
			filtered_ip0.append(self.__get_filtered_trace(self.regrouped_traces[current_s_box][g][0][i], filter_index))
			filtered_ip1.append(self.__get_filtered_trace(self.regrouped_traces[current_s_box][g][1][i], filter_index))
		return filtered_ip0, filtered_ip1

	def filter_traces(self, sbox_number):
		trace_matrix = numpy.zeros((256, self.nb_samples), dtype=numpy.int8)
		for i in range(256):
			trace_matrix [i] =self.__read_trace(self.traces[sbox_number][i])
		for i in range (self.nb_samples):
			if not numpy.unique(trace_matrix[:,i]).size == 1:
				self.filter_index[sbox_number].append(i)

	@staticmethod
	def is_disjoint_with_remove_consts(v1, v2):
		s1 = set(v1)
		s2 = set(v2)
		if len(s1) == 1 or len(s2) == 1:
			return False
		return not any(s1.intersection(s2))
	
	@staticmethod
	def is_disjoint(v1, v2):
		return not any(set(v1).intersection(set(v2)))
	
	def guess_key_chunk(self, current_s_box):
		score = 256*[0]
		start_time = time()
		print("target sbox-{} ...".format(current_s_box))
		for g in range(self.start_guess, self.end_guess):   # 0x10, 0x1f 0xd9, 0xf1
			start_filter_time = time()
			remain_samples = len(self.filter_index[current_s_box])
			ip0, ip1 = self.get_filtered_ip0_ip1(self.filter_index[current_s_box], current_s_box, g)
			for i in range(remain_samples):
				v1 = [ip0[j][i] for j in range(self.nb_traces_per_set)]
				v2 = [ip1[j][i] for j in range(self.nb_traces_per_set)]
				if self.is_disjoint(v1, v2):
					score[g] += 1
		best = [i for i, j in enumerate(score) if j == max(score)][0]
		print("sbox-{}: best = {} with {} disjoint-vectors, time {} sec."
			.format(current_s_box, hex(best), score[best], round(time() - start_time, 3)))
		if self.plot:
			plot_data(score)

		return best

	def guess_key_chunk_masked(self, current_s_box, window_inf, window_sup):
		score = 256*[0]
		start_time = time()
		print("target sbox-{} ...".format(current_s_box))
		for g in range(self.start_guess, self.end_guess):   # 0x10, 0x1f 0xd9, 0xf1
			start_filter_time = time()
			remain_samples = len(self.filter_index[current_s_box])
			ip0, ip1 = self.get_filtered_ip0_ip1(self.filter_index[current_s_box], current_s_box, g)
			for i in range(window_sup, remain_samples):
				for m in range (window_inf,window_sup):
					v1 = [ip0[j][i] ^ ip0[j][m] for j in range(self.nb_traces_per_set)]
					v2 = [ip1[j][i] ^ ip1[j][m] for j in range(self.nb_traces_per_set)]
					if self.is_disjoint(v1, v2):
						if not numpy.any(numpy.asarray(v1)) and not numpy.unique(v2).size == 1:
							score[g] += 1

		best = [i for i, j in enumerate(score) if j == max(score)]
		print("sbox-{}: best = {} with {} disjoint-vectors, time {} sec."
			.format(current_s_box, [hex (i) for i in best], score[best[0]], round(time() - start_time, 3)))
		return best[0]		
		
	def key_recovery(self, masked= False,  window=0, window_inf=0, window_sup=0):
		print("start round key recovery ...")
		print("traces path: {}".format(self.traces_root_path))
		start_attack_time = time()
		for i in range(self.start_s_box, self.end_s_box):
			if self.verbose:
				print("Number of simples per trace before filtering: {}.".format(self.nb_samples))
				print("Number of simples per trace after  filtering: {}.".format(len(self.filter_index[i])))
			if masked:
				#self.recovered_key[i] = self.guess_key_chunk_masked(i,  window) # window define where to find the mask
				self.recovered_key[i] = self.guess_key_chunk_masked(i, window_inf, window_sup)
			else:
				self.recovered_key[i] = self.guess_key_chunk(i)
		# if self.decrypt:
		#     self.recovered_key = reverse_key_schedule(self.recovered_key)
		self.master_key = reverse_key_schedule(self.recovered_key) if self.decrypt else self.recovered_key
		print("key recovery ({}) done in  {} sc.".
			format(bytes(self.master_key).hex(), round(time() - start_attack_time, 3)))
	