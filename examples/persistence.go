package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/darwayne/go-hnsw"
)

func main() {
	const (
		M              = 32
		efConstruction = 400
		efSearch       = 100
		K              = 10
		filename       = "persistence.example.hnsw.idx"
		totalVectors   = 3000
	)

	defer os.Remove(filename)

	var zero hnsw.Point = make([]float32, 128)

	h := hnsw.New(M, efConstruction, zero)
	h.Grow(totalVectors)

	queries := make([]hnsw.Point, 0, 1000)
	for i := 1; i <= totalVectors; i++ {
		point := randomPoint()
		if i <= 1000 {
			queries = append(queries, point)
		}

		h.AddPoint(point)
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}

	start := time.Now()
	h.Save(filename)
	t := time.Since(start)
	fmt.Println("It took", t, "to save to index")

	fmt.Printf("=== Generated Results (Next ID: %v) ===\n", h.NextID())
	search(h, queries, efSearch, K)

	start = time.Now()
	h, _, _ = hnsw.Load(filename)
	t = time.Since(start)
	fmt.Printf("=== Loaded Index Results (Next ID: %v) ===\n", h.NextID())
	fmt.Println("It took", t, "to load index")
	search(h, queries, efSearch, K)
}

func search(h *hnsw.Hnsw, queries []hnsw.Point, efSearch, K int) {
	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	truth := make([][]uint32, 1000)
	for i := range queries {
		result := h.SearchBrute(queries[i], K)
		truth[i] = make([]uint32, K)
		for j := K - 1; j >= 0; j-- {
			item := result.Pop()
			truth[i][j] = item.ID
		}
	}

	fmt.Printf("Now searching with HNSW...\n")
	hits := 0
	start := time.Now()
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

func randomPoint() hnsw.Point {
	var v hnsw.Point = make([]float32, 128)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
