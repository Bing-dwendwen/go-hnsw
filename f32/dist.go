package f32

func DistGo(a, b []float32) (r float32) {
	var d float32
	for i := range a {
		d = a[i] - b[i]
		r += d * d
	}
	return r
}
