package regression

import (
	"fmt"

	ab "github.com/oguzsozen/kisai/src/core/abstract"
	op "github.com/oguzsozen/kisai/src/core/algebra_operation"
	ex "github.com/oguzsozen/kisai/src/core/exception"
	ls "github.com/oguzsozen/kisai/src/loss_function"
	opt "github.com/oguzsozen/kisai/src/optimizer"
	tl "github.com/oguzsozen/kisai/src/tool"

	"gonum.org/v1/gonum/mat"
)

type RegressionModelMethod string

const (
	NORMAL_EQUATION RegressionModelMethod = "normal-equation"
	GRADIENT_DESCENT RegressionModelMethod = "gradient-descent"
)

type LRTrainConfig struct {
	IterationRate		int
	LearningRate		float64
	MemoryPrioritized	bool
	History				*tl.TrainHistory
}

type LinearRegression struct {
	*ab.FormulaBasedModel
	method	RegressionModelMethod
}

func NewLinearRegression(method RegressionModelMethod) *LinearRegression {
	return &LinearRegression{
		method: method,
		FormulaBasedModel: &ab.FormulaBasedModel{},
	}
}

func (lr *LinearRegression) fitNormalEquation(X *mat.Dense, y *mat.VecDense) {
	var xTXInverse mat.Dense
	var xTY mat.VecDense

	vecBias, _ := op.NewFillVecDense(y.Len(), 1)

	XBias, _ := op.AddColumnToMatrix(X, vecBias, 0)

	xTXInverse.Mul(XBias.T(), XBias)
	xTXInverse.Inverse(&xTXInverse)

	xTY.MulVec(XBias.T(), y)
	
	xTY.MulVec(&xTXInverse, &xTY)

	lr.updateWeightsAndBias(mat.VecDenseCopyOf(xTY.SliceVec(1, xTY.Len())), xTY.AtVec(0))
}

func (lr *LinearRegression) fitGradientDescent(X *mat.Dense, y *mat.VecDense, config *LRTrainConfig) {
	config.History = tl.NewTrainHistory()
	for i := 0; i < config.IterationRate; i++ {
		config.History.AddNewEpoch()

		iterateErr := mat.NewVecDense(y.Len(), nil)
		yPredicts := lr.internalPredictAll(X)

		iterateErr.SubVec(yPredicts, y)

		opt.GradientDescent(lr.FormulaBasedModel, X, iterateErr, config.LearningRate)

		config.History.AddNewMetric("mse", ls.MSEPreCalculated(iterateErr))
		
		fmt.Println(config.History.Epochs[len(config.History.Epochs) - 1])
	}
}


func (lr *LinearRegression) updateWeightsAndBias(w *mat.VecDense, b float64) {
	lr.Weight.CloneFromVec(w)
	lr.Bias = b
}

func (lr *LinearRegression) Fit(X *mat.Dense, y *mat.VecDense, config *LRTrainConfig) error {
	row, col := X.Dims()
	if row != y.Len() {
		return ex.ErrUnmatchDataLen
	}
	
	if row == 0 || col == 0 {
		return ex.ErrEmptyDataset
	}

	if !lr.checkTrained() {
		lr.Weight = mat.NewVecDense(col, nil)
	}

	switch lr.method {
		case NORMAL_EQUATION:
			lr.fitNormalEquation(X, y)
		case GRADIENT_DESCENT:
			lr.fitGradientDescent(X, y, config)
		default:
			return ex.ErrUnvalidMethod
	}

	return nil
}

func (lr *LinearRegression) Predict(x *mat.VecDense) (float64, error) {
	if !lr.checkTrained() {
		return 0, ex.ErrUntrainedModel
	}
	return lr.internalPredict(x), ex.ErrUntrainedModel
}

func (lr *LinearRegression) PredictAll(X *mat.Dense) (*mat.VecDense, error) {
	if !lr.checkTrained() {
		return nil, ex.ErrUntrainedModel
	}
	return lr.internalPredictAll(X), nil
}

func (lr *LinearRegression) internalPredictAll(X *mat.Dense) (*mat.VecDense) {
	r, _ := X.Dims()
	
	yPredicts := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		yPredicts.SetVec(i, mat.Dot(mat.VecDenseCopyOf(X.RowView(i)), lr.Weight) + lr.Bias)
	}
	
	return yPredicts
}

func (lr *LinearRegression) internalPredict(x *mat.VecDense) (float64) {
	return mat.Dot(x, lr.Weight) + lr.Bias
}

func (lr *LinearRegression) checkTrained() bool {
	return lr.Weight != nil
}