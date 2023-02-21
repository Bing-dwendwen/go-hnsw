package exmaples

import (
	"fmt"
	"github.com/Bing-dwendwen/go-hnsw"
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

	var zero gohnsw.Point = make([]float32, 128)

	h := gohnsw.New(M, efConstruction, zero, "")
	h.Grow(10000)

	start := time.Now()
	rand.Seed(time.Now().Unix())
	for i := 1; i <= 10000; i++ {
		h.AddPoint(randomPoint(), rand.Uint32())
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	queries := make([]gohnsw.Point, 1000)
	truth := make([][]uint32, 1000)
	for i := range queries {
		queries[i] = randomPoint()
		result := h.SearchBrute(queries[i], K)
		truth[i] = make([]uint32, K)
		for j := K - 1; j >= 0; j-- {
			item := result.Pop()
			truth[i][j] = item.ID
		}
	}

	fmt.Printf("Now searching with HNSW...\n")
	hits := 0
	start = time.Now()
	for i := 0; i < 1000; i++ {
		result := h.Search(queries[i], efSearch, K)
		for j := 0; j < K; j++ {
			item := result.Pop()
			for k := 0; k < K; k++ {
				if item.ID == truth[i][k] {
					hits++
				}
			}
		}
	}
	stop := time.Since(start)

	fmt.Printf("%v queries / second (single thread)\n", 1000.0/stop.Seconds())
	fmt.Printf("Average 10-NN precision: %v\n", float64(hits)/(1000.0*float64(K)))
}
