package algebraoperation

import (
	"math"

	exception "github.com/oguzsozen/kisai/src/core/exception"
	"gonum.org/v1/gonum/mat"
)

func SumOfSquaredDifference(a, b *mat.VecDense, memoryPrioritized bool) (float64, error) {
	if memoryPrioritized {
		return sumOfSquaredDifferenceMemoryEfficient(a, b)
	}
	return sumOfSquaredDifferenceTimeEfficient(a, b)
}

func sumOfSquaredDifferenceTimeEfficient(a, b *mat.VecDense) (float64, error) {
	c := mat.NewVecDense(a.Len(), nil)
	c.SubVec(a, b)
	c.MulElemVec(c, c)
	return mat.Sum(c), nil
}

func sumOfSquaredDifferenceMemoryEfficient(a, b *mat.VecDense) (float64, error) {
	var sum float64

	if a.Len() != b.Len() {
		return sum, exception.ErrUnmatchDataLen
	}

	for i := 0; i < a.Len(); i++ {
		sum += math.Pow(a.AtVec(i) - b.AtVec(i), 2)
	}

	return sum, nil
}

func AddConstToVec(a *mat.VecDense, v float64) {
	for i := 0; i < a.Len(); i++ {
		a.SetVec(i, a.AtVec(i) + v)
	}
}

func AddConstToNewVec(a *mat.VecDense, v float64) *mat.VecDense {
	data := a.RawVector().Data

	for i := 0; i < len(data); i++ {
		data[i] += v
	}

	return mat.NewVecDense(len(data), data)
}

func FillVecDense(vec *mat.VecDense, v float64) {
	for i := 0; i < vec.Len(); i++ {
		vec.SetVec(i, v)
	}
}

func NewFillVecDense(n int, v float64) (*mat.VecDense, error) {
	if n <= 0 {
		return nil, nil
	}

	vec := make([]float64, n)
	for i := 0; i < n; i++ {
		vec[i] = v
	}

	return mat.NewVecDense(n, vec), nil
}

func SubVecToVec(a, b *mat.VecDense) {
	for i := 0; i < a.Len(); i++ {
		a.SetVec(i, a.AtVec(i) - b.AtVec(i))
	}
}

func SubVecToNewVec(a, b *mat.VecDense) *mat.VecDense {
	c := mat.NewVecDense(a.Len(), nil)

	for i := 0; i < a.Len(); i++ {
		c.SetVec(i, a.AtVec(i) - b.AtVec(i))
	}

	return c
}

func AddVecToVec(a, b *mat.VecDense) {
	for i := 0; i < a.Len(); i++ {
		a.SetVec(i, a.AtVec(i) + b.AtVec(i))
	}
}

func AddVecToNewVec(a, b *mat.VecDense) *mat.VecDense {
	c := mat.NewVecDense(a.Len(), nil)

	for i := 0; i < a.Len(); i++ {
		c.SetVec(i, a.AtVec(i) + b.AtVec(i))
	}

	return c
}