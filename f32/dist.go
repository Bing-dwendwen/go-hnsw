package f32

import "math"

func DistGo(a, b []float32) (r float32) {
	var d float32
	for i := range a {
		d = a[i] - b[i]
		r += d * d
	}
	return r
}

// CalNormalizedCos 已经归一化的向量
func CalNormalizedCos(x, y []float32) float32 {
	var dotProduct float32
	for i, j := range y {
		dotProduct += x[i] * j
	}
	return 1 - dotProduct
}

func CalCos(x, y []float32) float32 {
	var dotProduct, xMagnitude, yMagnitude float32
	for i, j := range y {
		dotProduct += x[i] * j
		xMagnitude += x[i] * x[i]
		yMagnitude += j * j
	}
	xMagnitude = float32(math.Sqrt(float64(xMagnitude)))
	yMagnitude = float32(math.Sqrt(float64(yMagnitude)))
	return 1 - dotProduct/(xMagnitude*yMagnitude)
}
