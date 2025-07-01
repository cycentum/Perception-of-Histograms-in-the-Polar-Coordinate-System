def calc_binWidth(bins):
# 	return bins[1:]-bins[:-1]
	return bins.width


def calc_binCenter(bins):
# 	return (bins[1:]+bins[:-1])/2
	return bins.center


def normalizeDensity(distribution, bins):
	binWidth=calc_binWidth(bins)
	density=distribution/distribution.sum()/binWidth
	return density

