package optimizer

import (
	ab "github.com/oguzsozen/kisai/src/core/abstract"
	op "github.com/oguzsozen/kisai/src/core/algebra_operation"
	"gonum.org/v1/gonum/mat"
)

func GradientDescent(m *ab.FormulaBasedModel, X *mat.Dense, errRate *mat.VecDense, learningRate float64) {
	r, c := X.Dims()

	gWeight := mat.NewVecDense(c, nil)

	for j := 0; j < c; j++ {
		sum := 0.0
		for i := 0; i < r; i++ {
			sum += X.At(i, j) * errRate.AtVec(i)
		}
		gWeight.SetVec(j, sum / (2/float64(r)))
	}

	gBias := (2/float64(r)) * mat.Sum(errRate)

	updateGradient(m, gWeight, gBias, learningRate)
}

func updateGradient(m *ab.FormulaBasedModel, gWeight *mat.VecDense, gBias float64, learningRate float64) {
	gWeight.ScaleVec(learningRate, gWeight)
	op.SubVecToVec(m.Weight, gWeight)
	m.Bias -= learningRate * gBias
}