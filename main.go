// Copyright 2024 The Gauss Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/matrix"
)

var (
	// FlagMark1 mark 1 model
	FlagMark1 = flag.Bool("mark1", false, "mark 1 model")
	// FlagMark2 mark 2 model
	FlagMark2 = flag.Bool("mark2", false, "mark 2 model")
	// FlagMark3 mark 3 model
	FlagMark3 = flag.Bool("mark3", false, "mark 3 model")
)

func MatrixInverse(m Matrix, rng *Rand) (ai Matrix) {
	identity := NewIdentityMatrix(m.Cols)
	s := Meta(512, 2.7e-1, .1, rng, 4, .1, 1, false, func(samples []Sample, x ...Matrix) {
		done := make(chan bool, 8)
		process := func(index int) {
			x := samples[index].Vars[0][0].Sample()
			y := samples[index].Vars[0][1].Sample()
			z := samples[index].Vars[0][2].Sample()
			ai := x.Add(y.H(z))
			cost := m.MulT(ai).Quadratic(identity).Avg()
			samples[index].Cost = float64(cost.Data[0])
			done <- true
		}
		for j := range samples {
			go process(j)
		}
		for range samples {
			<-done
		}
	}, m)
	return s.Vars[0][0].Sample().Add(s.Vars[0][1].Sample().H(s.Vars[0][2].Sample()))
}

func main() {
	flag.Parse()

	if *FlagMark1 {
		Mark1()
		return
	} else if *FlagMark2 {
		Mark2()
		return
	} else if *FlagMark3 {
		Mark3()
		return
	}

	plain := `You will rejoice to hear that no disaster has accompanied the
	commencement of an enterprise which you have regarded with such evil
	forebodings. I arrived here yesterday, and my first task is to assure
	my dear sister of my welfare and increasing confidence in the success
	of my undertaking.`
	r := rand.New(rand.NewSource(int64(1)))
	cipher := []byte{}
	key := []byte{}
	for i, s := range plain {
		key = append(key, uint8(r.Uint32()))
		cipher = append(cipher, uint8(s)^key[i])
	}

	rng := Rand(1)
	optimizer := NewOptimizer(&rng, 16, .1, 1, func(samples []Sample, x ...Matrix) {
		for index := range samples {
			x := samples[index].Vars[0][0].Sample()
			y := samples[index].Vars[0][1].Sample()
			z := samples[index].Vars[0][2].Sample()
			sample := x.Add(y.H(z))
			key := []byte{}
			plain := []byte{}
			for r := 0; r < len(cipher); r++ {
				s := byte(0)
				for c := 0; c < 8; c++ {
					s <<= 1
					if sample.Data[r*8+c] > 0 {
						s |= 1
					}
				}
				plain = append(plain, s^cipher[r])
				key = append(key, s)
			}
			input := NewMatrix(256, len(cipher))
			control := NewMatrix(256, len(cipher))
			for i, s := range plain {
				ss := key[i]
				a, b := make([]float32, 256), make([]float32, 256)
				a[s] = 1
				b[ss] = 1
				input.Data = append(input.Data, a...)
				control.Data = append(control.Data, b...)
			}
			e1 := SelfEntropy(input, input, input)
			e2 := SelfEntropy(control, control, control)
			cost := 0.0
			for i, e := range e1 {
				cost += float64(e / e2[i])
			}
			samples[index].Cost = cost
		}
	}, NewCoord(8, len(cipher)))
	last := -1.0
	dx := 1e-9
	for {
		s := optimizer.Iterate()
		if last > 0 && math.Abs(last-s.Cost) < dx {
			break
		}
		fmt.Println(s.Cost)
		last = s.Cost
		x := s.Vars[0][0].Sample()
		y := s.Vars[0][1].Sample()
		z := s.Vars[0][2].Sample()
		sample := x.Add(y.H(z))
		plain := []byte{}
		for r := 0; r < len(cipher); r++ {
			s := byte(0)
			for c := 0; c < 8; c++ {
				s <<= 1
				if sample.Data[r*8+c] > 0 {
					s |= 1
				}
			}
			plain = append(plain, s^cipher[r])
		}
		p := make([]rune, len(plain))
		for i, value := range plain {
			p[i] = rune(value)
		}
		fmt.Println(string(p))
	}
}

// Mark3 mark 3 model
func Mark3() {
	plain := `You will rejoice to hear that no disaster has accompanied the
	commencement of an enterprise which you have regarded with such evil
	forebodings. I arrived here yesterday, and my first task is to assure
	my dear sister of my welfare and increasing confidence in the success
	of my undertaking.`
	lt, gt := 0, 0
	for seed := 1; seed < 8*1024; seed++ {
		rng := rand.New(rand.NewSource(int64(seed)))
		cipher := []byte{}
		key := []byte{}
		for i, s := range plain {
			key = append(key, uint8(rng.Uint32()))
			cipher = append(cipher, uint8(s)^key[i])
		}
		input := NewMatrix(256, len(cipher))
		control := NewMatrix(256, len(cipher))
		for i, s := range cipher {
			ss := key[i]
			a, b := make([]float32, 256), make([]float32, 256)
			a[s] = 1
			b[ss] = 1
			input.Data = append(input.Data, a...)
			control.Data = append(control.Data, b...)
		}
		entropy := SelfEntropy(input, input, input)
		avg := float32(0)
		for _, e := range entropy {
			avg += e
		}
		avg /= float32(len(cipher))
		entropy = SelfEntropy(control, control, control)
		avg2 := float32(0)
		for _, e := range entropy {
			avg2 += e
		}
		avg2 /= float32(len(cipher))
		fmt.Println(avg, avg2)
		if avg > avg2 {
			gt++
		} else {
			lt++
		}
	}
	fmt.Println(gt, lt)
}

// Mark2 mark 2 model
func Mark2() {
	primes := []uint64{1, 2, 3}
search:
	for i := uint64(4); i < 258; i++ {
		max := uint64(math.Sqrt(float64(i)) + .5)
		for _, p := range primes {
			if p == 1 {
				continue
			}
			if p > max {
				break
			}
			if i%p == 0 {
				continue search
			}
		}
		primes = append(primes, i)
	}
	fmt.Println(primes)
	type Integer struct {
		Integer uint64
		Factors []float32
		Entropy float32
	}
	integers := make([]Integer, 258)
	for i := range integers {
		ii := uint64(i)
		integers[i].Integer = ii
		integers[i].Factors = make([]float32, len(primes))
		max := uint64(math.Sqrt(float64(i)) + .5)
		for j, p := range primes {
			if p > max {
				break
			}
			for ii%p == 0 {
				integers[i].Factors[j]++
				if p == 1 {
					break
				}
				ii /= p
			}
		}
	}
	vectors := NewMatrix(len(primes), len(integers))
	for _, integer := range integers {
		vectors.Data = append(vectors.Data, integer.Factors...)
	}
	entropy := SelfEntropy(vectors, vectors, vectors)
	for i := range integers {
		integers[i].Entropy = entropy[i]
	}
	sort.Slice(integers, func(i, j int) bool {
		return integers[i].Entropy < integers[j].Entropy
	})
	for _, integer := range integers {
		fmt.Println(integer.Factors, integer.Integer, integer.Entropy)
	}
}

// Mark1 mark 1 model
func Mark1() {
	//rng := Rand(1)
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	type Row struct {
		Score    float64
		Measures []float64
		Label    string
	}
	rows := make([]Row, len(datum.Fisher))
	for i := range rows {
		rows[i].Measures = datum.Fisher[i].Measures
		rows[i].Label = datum.Fisher[i].Label
	}
	split := func(rows []Row) int {
		points := NewMatrix(4, len(rows))
		for _, row := range rows {
			for _, value := range row.Measures {
				points.Data = append(points.Data, float32(value))
			}
		}
		entropy := SelfEntropy64(points, points, points)
		for i, e := range entropy {
			rows[i].Score = e
		}
		multi := NewMultiFromData(points.T())
		sort.Slice(rows, func(i, j int) bool {
			return rows[i].Score < rows[j].Score
		})
		in := float32(0.0)
		for i := 0; i < len(multi.E.Data); i++ {
			x := multi.E.Data[i]
			if x < 0 {
				x = -x
			}
			in += x
		}
		max, index := float32(0.0), 0
		for i := 0; i < len(rows)-1; i++ {
			a := NewMatrix(4, i+1)
			b := NewMatrix(4, len(rows)-(i+1))
			for j := 0; j < i+1; j++ {
				row := rows[j]
				for _, value := range row.Measures {
					a.Data = append(a.Data, float32(value))
				}
			}
			for j := i + 1; j < len(rows); j++ {
				row := rows[j]
				for _, value := range row.Measures {
					b.Data = append(b.Data, float32(value))
				}
			}
			aa := NewMultiFromData(a.T())
			aaa := float32(0.0)
			for i := 0; i < len(aa.E.Data); i++ {
				x := aa.E.Data[i]
				if x < 0 {
					x = -x
				}
				aaa += x
			}
			bb := NewMultiFromData(b.T())
			bbb := float32(0.0)
			for i := 0; i < len(bb.E.Data); i++ {
				x := bb.E.Data[i]
				if x < 0 {
					x = -x
				}
				bbb += x
			}
			if gap := in - (aaa + bbb); gap > max {
				max, index = gap, i
			}
		}
		return index
	}
	index := split(rows)
	fmt.Println(index)
	for i := 0; i <= index; i++ {
		fmt.Println(rows[i].Label, rows[i].Score, rows[i].Measures)
	}
	rows = rows[index+1:]
	index = split(rows)
	fmt.Println(index)
	for i := 0; i <= index; i++ {
		fmt.Println(rows[i].Label, rows[i].Score, rows[i].Measures)
	}
	fmt.Println(index)
	for i := index + 1; i < len(rows); i++ {
		fmt.Println(rows[i].Label, rows[i].Score, rows[i].Measures)
	}
	/*multi.LearnA(&rng, nil)
	fmt.Println(multi.A)
	inv := MatrixInverse(multi.A, &rng)
	fmt.Println(inv)
	fmt.Println(multi.A.MulT(inv))
	dist := inv.MulT(points.Sub(multi.U))
	fmt.Println(dist)*/
}
