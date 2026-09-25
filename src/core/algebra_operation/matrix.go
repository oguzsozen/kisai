package algebraoperation

import (
	ex "github.com/oguzsozen/kisai/src/core/exception"
	"gonum.org/v1/gonum/mat"
)

func AddConstToMatrix(a *mat.Dense, v float64) {
	a.Apply(func(i, j int, value float64) float64 {
		return value + v
	}, a)
}

func AddConstToNewMatrix(a *mat.Dense, v float64) *mat.Dense {
	r, c := a.Dims()
	b := mat.NewDense(r, c, nil)

	b.Apply(func(i, j int, value float64) float64 {
		return value + v
	}, a)

	return b
}

func AddColumnToMatrix(a *mat.Dense, b *mat.VecDense, j int) (*mat.Dense, error) {
	r, c := a.Dims()

	if r != b.Len() {
		return nil, ex.ErrUnmatchDataLen
	}

	if j < 0 || j > c {
		return nil, ex.ErrOutOfRange
	}

	temp := mat.NewDense(r, c + 1, nil)
	switch j {
		case 0:
			temp.Slice(0, r, 1, c + 1).(*mat.Dense).Copy(a)
			temp.ColView(0).(*mat.VecDense).CopyVec(b)
			return temp, nil
		case c:
			temp.Slice(0, r, 0, c).(*mat.Dense).Copy(a)
			temp.ColView(c).(*mat.VecDense).CopyVec(b)
			return temp, nil
		default:
			temp.Slice(0, r, 0, j).(*mat.Dense).Copy(a.Slice(0, r, 0, j))
			temp.ColView(j).(*mat.VecDense).CopyVec(b)
			temp.Slice(0, r, j + 1, c + 1).(*mat.Dense).Copy(a.Slice(0, r, j, c))
			return temp, nil
	}
}