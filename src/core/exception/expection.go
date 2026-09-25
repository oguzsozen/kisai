package exception

import (
	"fmt"
)

type Expection struct {
	message	string
	detail	string
}

func newException(message, detail string) *Expection {
	return &Expection{message: message, detail: detail}
}

func (e *Expection) Error() string {
	return fmt.Sprintf("[%s]: %s", e.message, e.detail)
}

var (
	ErrUntrainedModel = newException("Untrained Model Error", "The model never trained yet. Use Fit() for training model.")
	ErrUnmatchDataLen = newException("Unmatch Data Lenght Error", "Both data must have same lenght.")
	ErrEmptyDataset = newException("Empty Dataset Error", "Dataset can not be empty.")
	ErrUnvalidMethod = newException("Unvalid Method Error", "Used method is not valid.")
	ErrOutOfRange = newException("Out Of Range Error", "The index is not in accessible range.")
)