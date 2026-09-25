package lossfunction

import (
	op "github.com/oguzsozen/kisai/src/core/algebra_operation"
	"gonum.org/v1/gonum/mat"
)

func MSE(y, yPredicts *mat.VecDense) (float64, error) {
	sum, err := op.SumOfSquaredDifference(y, yPredicts, false)
	if err != nil {
		return 0, err
	}
	
	return sum / float64(y.Len()), nil
}

func MSEPreCalculated(errRate *mat.VecDense) float64 {
	return mat.Sum(errRate) / float64(errRate.Len())
}