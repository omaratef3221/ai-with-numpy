import numpy as np

def positional_encoding(position: int, d_model: int):
	"""
	Positional Encoding computed only once and doesn't recompute during training or inf again
	It has two signales
	power = 10000^((2*i)/d_model)
	PE(pos_at_seq, (2*idx_embd)) = Sin(pos_at_seq/power)
	PE(pos_at_seq, (2*idx_embd) + 1) = Cos(pos_at_seq/power)
	"""
	if d_model <= 0 or position == 0:
		return -1
		
	embds = np.zeros((position, d_model))
	for pos in range(position):
		for idx in np.arange(int(d_model/2)):
			power = np.power(10000, (2*idx)/d_model)
			embds[pos][2*idx] = np.sin(pos/power)
			embds[pos][2*idx+1] = np.cos(pos/power)

	return np.round(embds,4)