package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {

	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 10
	)

	var zero Point = make([]float32, 128)

	h := New(M, efConstruction, zero)
	h.Grow(10000)

	// Note that added ID:s must start from 1
	for i := 1; i <= 10000; i++ {
		h.Add(randomPoint(), uint32(i))
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}

	start := time.Now()
	for i := 0; i < 1000; i++ {
		h.Search(randomPoint(), efSearch, K)
	}
	stop := time.Since(start)

	fmt.Printf("%v queries / second (single thread)\n", 1000.0/stop.Seconds())
}

func randomPoint() Point {
	var v Point = make([]float32, 128)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
