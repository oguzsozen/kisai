package metric

import (
	"math"

	op "github.com/oguzsozen/kisai/src/core/algebra_operation"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func R2Score(y, yPredicts *mat.VecDense) (float64, error) {
	var ssr, sst float64
	mean := stat.Mean(y.RawVector().Data, nil)

	ssr, err := op.SumOfSquaredDifference(y, yPredicts, false)
	if err != nil {
		return 0, err
	}

	

	for i := 0; i < y.Len(); i++ {
		sst += math.Pow(y.AtVec(i) - mean, 2)
	}

	if sst == 0 {
		return 0, nil
	}

	return 1 - (ssr / sst), nil
}