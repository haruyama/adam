package adam

import "math"

const (
	DefaultAlpha  = 0.001
	DefaultBeta1  = 0.9
	DefaultBeta2  = 0.999
	DefaultError  = 0.00000001
	DefaultLambda = 0.99999999

	PositiveLevel = 1
	NegativeLevel = -1
	Margin        = 1.0
)

type Adam struct {
	Alpha  float64
	Beta1  float64
	Beta2  float64
	Error  float64
	Lambda float64

	Beta1T float64
	Beta1P float64
	Beta2P float64

	Weight  map[string]float64
	Moment1 map[string]float64
	Moment2 map[string]float64
}

func NewAdam() *Adam {
	return &Adam{
		Alpha:  DefaultAlpha,
		Beta1:  DefaultBeta1,
		Beta2:  DefaultBeta2,
		Error:  DefaultError,
		Lambda: DefaultLambda,

		Beta1T: DefaultBeta1,
		Beta1P: DefaultBeta1,
		Beta2P: DefaultBeta2,

		Weight:  map[string]float64{},
		Moment1: map[string]float64{},
		Moment2: map[string]float64{},
	}
}

func (a *Adam) margin(data map[string]float64) float64 {
	margin := 0.0
	for f, v := range data {
		if a.Weight[f] == 0 {
			continue
		}
		margin += a.Weight[f] * v
	}
	return margin
}

func (a *Adam) Classify(data map[string]float64) int {
	if a.margin(data) > 0 {
		return PositiveLevel
	}
	return NegativeLevel
}

func (a *Adam) Update(label int, data map[string]float64) {
	if float64(label)*a.margin(data) > Margin {
		return
	}

	for f, v := range data {
		if v == 0 {
			continue
		}
		gradient := -1.0 * float64(label) * v

		a.Moment1[f] *= a.Beta1T
		a.Moment1[f] += (1.0 - a.Beta1T) * gradient

		a.Moment2[f] *= a.Beta2
		a.Moment2[f] += (1.0 - a.Beta2) * gradient * gradient

		a.Weight[f] -= a.Alpha * math.Sqrt(1.0-a.Beta2P) / (1.0 - a.Beta1P) * a.Moment1[f] / (math.Sqrt(a.Moment2[f]) + a.Error)
	}
	a.Beta1T *= a.Lambda
	a.Beta1P *= a.Beta1
	a.Beta2P *= a.Beta2
}
