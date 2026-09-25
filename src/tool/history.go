package tool

import (
	"fmt"
	"time"
)

type TrainHistory struct {
	DateTime	time.Time
	Epochs		[]HistoryEpoch
}

type HistoryEpoch struct {
	EpochNumber	int
	Metrics		[]MetricValue
}

type MetricValue struct {
	Name	string
	Value	float64
}

func NewTrainHistory() *TrainHistory {
	return &TrainHistory{
		DateTime: time.Now(),
		Epochs: []HistoryEpoch{},
	}
}

func (th *TrainHistory) AddNewEpoch() {
	th.Epochs = append(th.Epochs, HistoryEpoch{
		EpochNumber: len(th.Epochs) + 1,
		Metrics: []MetricValue{},
	})
}

func (th *TrainHistory) AddNewMetric(name string, value float64) {
	th.Epochs[len(th.Epochs) - 1].Metrics = append(
		th.Epochs[len(th.Epochs) - 1].Metrics,
		MetricValue{
			Name: name,
			Value: value,
		})
}

func (he HistoryEpoch) String() string {
	metrics := ""
	for i := 0; i < len(he.Metrics); i++ {
		metrics += fmt.Sprintf("%v | ", he.Metrics[i])
	}
	metrics = metrics[:len(metrics) - 4]
	return fmt.Sprintf("Epoch: %d\n\t%s", he.EpochNumber, metrics)
}

func (mv MetricValue) String() string {
	return fmt.Sprintf("%s: %f", mv.Name, mv.Value)
}