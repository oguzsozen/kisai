package abstract

import "gonum.org/v1/gonum/mat"

type baseModel struct{
	
}

type DataBasedModel struct {
	baseModel
}

type FormulaBasedModel struct {
	baseModel
	Weight *mat.VecDense
	Bias   float64
}