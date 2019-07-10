from bucketing.core.aes import Bucketing


TRACES_ROOT_PATH = "./traces/"

bucket = Bucketing(TRACES_ROOT_PATH, start_s_box=0, end_s_box=16, decrypt=False, plot=False, verbose=False)

bucket.key_recovery(masked=True, window_inf=0, window_sup=100)
