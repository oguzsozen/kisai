package main

import (
	"fmt"
	"os"

	metric "github.com/oguzsozen/kisai/src/metric"
	reg "github.com/oguzsozen/kisai/src/regression"
	"gonum.org/v1/gonum/mat"
)

func main() {
	trainX := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	trainy := mat.NewVecDense(5, []float64{3, 5, 7, 9, 11})

	testX := mat.NewDense(5, 1, []float64{6, 7, 8, 9, 10})
	testy := mat.NewVecDense(5, []float64{13, 15, 17, 19, 21})

	// model := reg.NewLinearRegression(reg.GRADIENT_DESCENT)
	model := reg.NewLinearRegression(reg.NORMAL_EQUATION)

	fmt.Println("Model created...")

	config := &reg.LRTrainConfig{
		IterationRate: 1000,
		LearningRate: 0.01,
		MemoryPrioritized: false,
	}

	if err := model.Fit(trainX, trainy, config); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println("Model trained...")

	testyPredicts, err := model.PredictAll(testX)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println("Model evaluating...")

	fmt.Println(testyPredicts)

	score, err := metric.R2Score(testy, testyPredicts)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("R2 Score: %.2f", score)

}