package activationfunction

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}