package hnsw

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/darwayne/go-hnsw/bitsetpool"
	"github.com/darwayne/go-hnsw/distqueue"
	"github.com/darwayne/go-hnsw/f32"

	"golang.org/x/sys/cpu"
)

var useAVX2 bool
var useAVX bool
var useSSE4 bool

type DistanceFunction func([]float32, []float32) float32

var DefaultDistFunc DistanceFunction

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useAVX = cpu.X86.HasAVX
	useSSE4 = cpu.X86.HasSSE41

	switch {
	case useAVX, useAVX2:
		log.Println("AVX instruction set detected")
		DefaultDistFunc = f32.L2Squared8AVX
	case useSSE4:
		log.Println("SSE instruction set detected")
		DefaultDistFunc = f32.L2Squared
	default:
		log.Println("No CPU vector extensions detected")
		DefaultDistFunc = f32.DistGo
	}
}

type Point []float32

func (a Point) Size() int {
	return len(a) * 4
}

type node struct {
	sync.RWMutex
	locked  bool
	p       Point
	level   int
	friends [][]uint32
}

type Hnsw struct {
	sync.RWMutex
	M              int
	M0             int
	efConstruction int
	linkMode       int
	DelaunayType   int
	nextID         uint32

	DistFunc func([]float32, []float32) float32

	nodes []node

	bitset *bitsetpool.BitsetPool

	LevelMult  float64
	maxLayer   int
	enterpoint uint32
}

// Load opens a index file previously written by Save(). Returns a new index and the timestamp the file was written
func Load(filename string) (*Hnsw, int64, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, 0, err
	}

	timestamp := readInt64(z)

	h := new(Hnsw)
	h.M = readInt32(z)
	h.M0 = readInt32(z)
	h.efConstruction = readInt32(z)
	h.linkMode = readInt32(z)
	h.DelaunayType = readInt32(z)
	h.LevelMult = readFloat64(z)
	h.maxLayer = readInt32(z)
	h.enterpoint = uint32(readInt32(z))
	h.nextID = uint32(readInt32(z))

	h.DistFunc = DefaultDistFunc
	h.bitset = bitsetpool.New()

	l := readInt32(z)
	h.nodes = make([]node, l)

	for i := range h.nodes {

		l := readInt32(z)
		h.nodes[i].p = make([]float32, l)

		err = binary.Read(z, binary.LittleEndian, h.nodes[i].p)
		if err != nil {
			return nil, 0, err
		}
		h.nodes[i].level = readInt32(z)

		l = readInt32(z)
		h.nodes[i].friends = make([][]uint32, l)

		for j := range h.nodes[i].friends {
			l := readInt32(z)
			h.nodes[i].friends[j] = make([]uint32, l)
			err = binary.Read(z, binary.LittleEndian, h.nodes[i].friends[j])
			if err != nil {
				return nil, 0, err
			}
		}
	}

	if err = z.Close(); err != nil {
		return nil, 0, err
	}
	if err = f.Close(); err != nil {
		return nil, 0, err
	}

	return h, timestamp, nil
}

// Save writes to current index to a gzipped binary data file
func (h *Hnsw) Save(filename string) error {
	f, err := ioutil.TempFile("", "hnsw")
	if err != nil {
		return err
	}

	defer func() {
		if err != nil {
			os.Remove(f.Name())
		}
	}()

	z := gzip.NewWriter(f)

	timestamp := time.Now().Unix()

	writeInt64(timestamp, z)

	writeInt32(h.M, z)
	writeInt32(h.M0, z)
	writeInt32(h.efConstruction, z)
	writeInt32(h.linkMode, z)
	writeInt32(h.DelaunayType, z)
	writeFloat64(h.LevelMult, z)
	writeInt32(h.maxLayer, z)
	writeInt32(int(h.enterpoint), z)
	writeInt32(int(h.nextID), z)

	l := len(h.nodes)
	writeInt32(l, z)

	if err != nil {
		return err
	}
	for _, n := range h.nodes {
		l := len(n.p)
		writeInt32(l, z)
		err = binary.Write(z, binary.LittleEndian, []float32(n.p))
		if err != nil {
			return err
		}
		writeInt32(n.level, z)

		l = len(n.friends)
		writeInt32(l, z)
		for _, f := range n.friends {
			l := len(f)
			writeInt32(l, z)
			err = binary.Write(z, binary.LittleEndian, f)
			if err != nil {
				return err
			}
		}
	}

	if err = z.Close(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}

	if err = os.Rename(f.Name(), filename); err != nil {
		return err
	}

	return nil
}

func writeInt64(v int64, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}

func writeInt32(v int, w io.Writer) {
	i := int32(v)
	err := binary.Write(w, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
}

func readInt32(r io.Reader) int {
	var i int32
	err := binary.Read(r, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
	return int(i)
}

func writeFloat64(v float64, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}

func readInt64(r io.Reader) (v int64) {
	err := binary.Read(r, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
	return
}

func readFloat64(r io.Reader) (v float64) {
	err := binary.Read(r, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
	return
}

func (h *Hnsw) getFriends(n uint32, level int) []uint32 {
	if len(h.nodes[n].friends) < level+1 {
		return make([]uint32, 0)
	}
	return h.nodes[n].friends[level]
}

func (h *Hnsw) NextID() uint32 {
	return atomic.LoadUint32(&h.nextID)
}

func (h *Hnsw) Link(first, second uint32, level int) {
	maxL := h.M
	if level == 0 {
		maxL = h.M0
	}

	h.RLock()
	node := &h.nodes[first]
	h.RUnlock()

	node.Lock()

	// check if we have allocated friends slices up to this level?
	if len(node.friends) < level+1 {
		for j := len(node.friends); j <= level; j++ {
			// allocate new list with 0 elements but capacity maxL
			node.friends = append(node.friends, make([]uint32, 0, maxL))
		}
		// now grow it by one and add the first connection for this layer
		node.friends[level] = node.friends[level][0:1]
		node.friends[level][0] = second

	} else {
		// we did have some already... this will allocate more space if it overflows maxL
		node.friends[level] = append(node.friends[level], second)
	}

	l := len(node.friends[level])

	if l > maxL {

		// to many links, deal with it

		switch h.DelaunayType {
		case 0:
			resultSet := &distqueue.DistQueueClosestLast{Size: len(node.friends[level])}

			for _, n := range node.friends[level] {
				resultSet.Push(n, h.DistFunc(node.p, h.nodes[n].p))
			}
			for resultSet.Len() > maxL {
				resultSet.Pop()
			}
			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends[level] = node.friends[level][0:maxL]
			for i := maxL - 1; i >= 0; i-- {
				item := resultSet.Pop()
				node.friends[level][i] = item.ID
			}

		case 1:

			resultSet := &distqueue.DistQueueClosestFirst{Size: len(node.friends[level])}

			for _, n := range node.friends[level] {
				resultSet.Push(n, h.DistFunc(node.p, h.nodes[n].p))
			}
			h.getNeighborsByHeuristicClosestFirst(resultSet, maxL)

			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends[level] = node.friends[level][0:maxL]
			for i := 0; i < maxL; i++ {
				item := resultSet.Pop()
				node.friends[level][i] = item.ID
			}
		}
	}
	node.Unlock()
}

func (h *Hnsw) getNeighborsByHeuristicClosestLast(resultSet1 *distqueue.DistQueueClosestLast, M int) {
	if resultSet1.Len() <= M {
		return
	}
	resultSet := &distqueue.DistQueueClosestFirst{Size: resultSet1.Len()}
	tempList := &distqueue.DistQueueClosestFirst{Size: resultSet1.Len()}
	result := make([]*distqueue.Item, 0, M)
	for resultSet1.Len() > 0 {
		resultSet.PushItem(resultSet1.Pop())
	}
	for resultSet.Len() > 0 {
		if len(result) >= M {
			break
		}
		e := resultSet.Pop()
		good := true
		for _, r := range result {
			if h.DistFunc(h.nodes[r.ID].p, h.nodes[e.ID].p) < e.D {
				good = false
				break
			}
		}
		if good {
			result = append(result, e)
		} else {
			tempList.PushItem(e)
		}
	}
	for len(result) < M && tempList.Len() > 0 {
		result = append(result, tempList.Pop())
	}
	for _, item := range result {
		resultSet1.PushItem(item)
	}
}

func (h *Hnsw) getNeighborsByHeuristicClosestFirst(resultSet *distqueue.DistQueueClosestFirst, M int) {
	if resultSet.Len() <= M {
		return
	}
	tempList := &distqueue.DistQueueClosestFirst{Size: resultSet.Len()}
	result := make([]*distqueue.Item, 0, M)
	for resultSet.Len() > 0 {
		if len(result) >= M {
			break
		}
		e := resultSet.Pop()
		good := true
		for _, r := range result {
			if h.DistFunc(h.nodes[r.ID].p, h.nodes[e.ID].p) < e.D {
				good = false
				break
			}
		}
		if good {
			result = append(result, e)
		} else {
			tempList.PushItem(e)
		}
	}
	for len(result) < M && tempList.Len() > 0 {
		result = append(result, tempList.Pop())
	}
	resultSet.Reset()

	for _, item := range result {
		resultSet.PushItem(item)
	}
}

func New(M int, efConstruction int, first Point) *Hnsw {
	h := Hnsw{}
	h.M = M
	// default values used in c++ implementation
	h.LevelMult = 1 / math.Log(float64(M))
	h.efConstruction = efConstruction
	h.M0 = 2 * M
	h.DelaunayType = 1

	h.bitset = bitsetpool.New()

	h.DistFunc = DefaultDistFunc

	// add first point, it will be our enterpoint (index 0)
	h.nodes = []node{{level: 0, p: first}}

	return &h
}

func (h *Hnsw) Stats() string {
	var str strings.Builder
	str.WriteString("HNSW Index\n")
	str.WriteString(fmt.Sprintf("M: %v, efConstruction: %v\n", h.M, h.efConstruction))
	str.WriteString(fmt.Sprintf("DelaunayType: %v\n", h.DelaunayType))
	str.WriteString(fmt.Sprintf("Number of nodes: %v\n", len(h.nodes)))
	str.WriteString(fmt.Sprintf("Max layer: %v\n", h.maxLayer))
	memoryUseData := 0
	memoryUseIndex := 0
	levCount := make([]int, h.maxLayer+1)
	conns := make([]int, h.maxLayer+1)
	connsC := make([]int, h.maxLayer+1)
	for i := range h.nodes {
		levCount[h.nodes[i].level]++
		for j := 0; j <= h.nodes[i].level; j++ {
			if len(h.nodes[i].friends) > j {
				l := len(h.nodes[i].friends[j])
				conns[j] += l
				connsC[j]++
			}
		}
		memoryUseData += h.nodes[i].p.Size()
		memoryUseIndex += h.nodes[i].level*h.M*4 + h.M0*4
	}
	for i := range levCount {
		avg := conns[i] / max(1, connsC[i])
		str.WriteString(fmt.Sprintf("Level %v: %v nodes, average number of connections %v\n", i, levCount[i], avg))
	}
	str.WriteString(fmt.Sprintf("Memory use for data: %v (%v bytes / point)\n", memoryUseData, memoryUseData/len(h.nodes)))
	str.WriteString(fmt.Sprintf("Memory use for index: %v (avg %v bytes / point)\n", memoryUseIndex, memoryUseIndex/len(h.nodes)))

	return str.String()
}

func (h *Hnsw) Grow(size int) {
	if size+1 <= len(h.nodes) {
		return
	}
	newNodes := make([]node, len(h.nodes), size+1)
	copy(newNodes, h.nodes)
	h.nodes = newNodes
}

func (h *Hnsw) AddPoint(q Point) uint32 {
	id := atomic.AddUint32(&h.nextID, 1)
	h.Add(q, id)
	return id
}

func (h *Hnsw) Add(q Point, id uint32) {
	if id == 0 {
		panic("Id 0 is reserved, use ID:s starting from 1 when building index")
	}

	// generate random level
	curlevel := int(math.Floor(-math.Log(rand.Float64() * h.LevelMult)))

	epID := h.enterpoint
	currentMaxLayer := h.nodes[epID].level
	ep := &distqueue.Item{ID: h.enterpoint, D: h.DistFunc(h.nodes[h.enterpoint].p, q)}

	// assume Grow has been called in advance
	newID := id
	newNode := node{p: q, level: curlevel, friends: make([][]uint32, min(curlevel, currentMaxLayer)+1)}

	// first pass, find another ep if curlevel < maxLayer
	for level := currentMaxLayer; level > curlevel; level-- {
		changed := true
		for changed {
			changed = false
			for _, i := range h.getFriends(ep.ID, level) {
				d := h.DistFunc(h.nodes[i].p, q)
				if d < ep.D {
					ep = &distqueue.Item{ID: i, D: d}
					changed = true
				}
			}
		}
	}

	// second pass, ef = efConstruction
	// loop through every level from the new nodes level down to level 0
	// create new connections in every layer
	for level := min(curlevel, currentMaxLayer); level >= 0; level-- {

		resultSet := &distqueue.DistQueueClosestLast{}
		h.searchAtLayer(q, resultSet, h.efConstruction, ep, level)
		switch h.DelaunayType {
		case 0:
			// shrink resultSet to M closest elements (the simple heuristic)
			for resultSet.Len() > h.M {
				resultSet.Pop()
			}
		case 1:
			h.getNeighborsByHeuristicClosestLast(resultSet, h.M)
		}
		newNode.friends[level] = make([]uint32, resultSet.Len())
		for i := resultSet.Len() - 1; i >= 0; i-- {
			item := resultSet.Pop()
			// store in order, closest at index 0
			newNode.friends[level][i] = item.ID
		}
	}

	h.Lock()
	// Add it and increase slice length if necessary
	if len(h.nodes) < int(newID)+1 {
		h.nodes = h.nodes[0 : newID+1]
	}
	h.nodes[newID] = newNode
	h.Unlock()

	// now add connections to newNode from newNodes neighbours (makes it visible in the graph)
	for level := min(curlevel, currentMaxLayer); level >= 0; level-- {
		for _, n := range newNode.friends[level] {
			h.Link(n, newID, level)
		}
	}

	h.Lock()
	if curlevel > h.maxLayer {
		h.maxLayer = curlevel
		h.enterpoint = newID
	}
	h.Unlock()
}

func (h *Hnsw) searchAtLayer(q Point, resultSet *distqueue.DistQueueClosestLast, efConstruction int, ep *distqueue.Item, level int) {
	var pool, visited = h.bitset.Get()
	//visited := make(map[uint32]bool)

	candidates := &distqueue.DistQueueClosestFirst{Size: efConstruction * 3}

	visited.Set(uint(ep.ID))
	//visited[ep.ID] = true
	candidates.Push(ep.ID, ep.D)

	resultSet.Push(ep.ID, ep.D)

	for candidates.Len() > 0 {
		_, lowerBound := resultSet.Top() // worst distance so far
		c := candidates.Pop()

		if c.D > lowerBound {
			// since candidates is sorted, it wont get any better...
			break
		}

		if len(h.nodes[c.ID].friends) >= level+1 {
			friends := h.nodes[c.ID].friends[level]
			for _, n := range friends {
				if !visited.Test(uint(n)) {
					visited.Set(uint(n))
					d := h.DistFunc(q, h.nodes[n].p)
					_, topD := resultSet.Top()
					if resultSet.Len() < efConstruction {
						item := resultSet.Push(n, d)
						candidates.PushItem(item)
					} else if topD > d {
						// keep length of resultSet to max efConstruction
						item := resultSet.PopAndPush(n, d)
						candidates.PushItem(item)
					}
				}
			}
		}
	}
	h.bitset.Free(pool)
}

// SearchBrute returns the true K nearest neigbours to search point q
func (h *Hnsw) SearchBrute(q Point, K int) *distqueue.DistQueueClosestLast {
	resultSet := &distqueue.DistQueueClosestLast{Size: K}
	for i := 1; i < len(h.nodes); i++ {
		d := h.DistFunc(h.nodes[i].p, q)
		if resultSet.Len() < K {
			resultSet.Push(uint32(i), d)
			continue
		}
		_, topD := resultSet.Head()
		if d < topD {
			resultSet.PopAndPush(uint32(i), d)
			continue
		}
	}
	return resultSet
}

// Benchmark test precision by comparing the results of SearchBrute and Search
func (h *Hnsw) Benchmark(q Point, ef int, K int) float64 {
	result := h.Search(q, ef, K)
	groundTruth := h.SearchBrute(q, K)
	truth := make([]uint32, 0)
	for groundTruth.Len() > 0 {
		truth = append(truth, groundTruth.Pop().ID)
	}
	p := 0
	for result.Len() > 0 {
		i := result.Pop()
		for j := 0; j < K; j++ {
			if truth[j] == i.ID {
				p++
			}
		}
	}
	return float64(p) / float64(K)
}

func (h *Hnsw) Search(q Point, ef int, K int) *distqueue.DistQueueClosestLast {
	h.RLock()
	currentMaxLayer := h.maxLayer
	ep := &distqueue.Item{ID: h.enterpoint, D: h.DistFunc(h.nodes[h.enterpoint].p, q)}
	h.RUnlock()

	resultSet := &distqueue.DistQueueClosestLast{Size: ef + 1}
	// first pass, find best ep
	for level := currentMaxLayer; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			for _, i := range h.getFriends(ep.ID, level) {
				d := h.DistFunc(h.nodes[i].p, q)
				if d < ep.D {
					ep.ID, ep.D = i, d
					changed = true
				}
			}
		}
	}
	h.searchAtLayer(q, resultSet, ef, ep, 0)

	for resultSet.Len() > K {
		resultSet.Pop()
	}
	return resultSet
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
