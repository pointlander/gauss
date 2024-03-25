// Copyright 2024 The Gauss Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"sort"

	"github.com/pointlander/datum/iris"
	. "github.com/pointlander/matrix"
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
	//rng := Rand(1)
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	points := NewMatrix(4, 150)
	for _, row := range datum.Fisher {
		for _, value := range row.Measures {
			points.Data = append(points.Data, float32(value))
		}
	}
	multi := NewMultiFromData(points.T())
	fmt.Println(multi.E)
	sort.Slice(datum.Fisher, func(i, j int) bool {
		return datum.Fisher[i].Measures[0] < datum.Fisher[j].Measures[0]
	})
	in := float32(0.0)
	for i := 0; i < 4; i++ {
		x := multi.E.Data[i]
		if x < 0 {
			x = -x
		}
		in += x
	}
	max, index := float32(0.0), 0
	for i := 0; i < len(datum.Fisher)-1; i++ {
		a := NewMatrix(4, i+1)
		b := NewMatrix(4, len(datum.Fisher)-(i+1))
		for j := 0; j < i+1; j++ {
			row := datum.Fisher[j]
			for _, value := range row.Measures {
				a.Data = append(a.Data, float32(value))
			}
		}
		for j := i + 1; j < len(datum.Fisher); j++ {
			row := datum.Fisher[j]
			for _, value := range row.Measures {
				b.Data = append(b.Data, float32(value))
			}
		}
		aa := NewMultiFromData(a.T())
		aaa := float32(0.0)
		for i := 0; i < 4; i++ {
			x := aa.E.Data[i]
			if x < 0 {
				x = -x
			}
			aaa += x
		}
		bb := NewMultiFromData(b.T())
		bbb := float32(0.0)
		for i := 0; i < 4; i++ {
			x := bb.E.Data[i]
			if x < 0 {
				x = -x
			}
			bbb += x
		}
		if gap := in - (aaa + bbb); gap > max {
			fmt.Println(i, in, aaa, bbb, gap)
			max, index = gap, i
		}
	}
	fmt.Println(index)
	for i := 0; i < index; i++ {
		fmt.Println(datum.Fisher[i].Label)
	}
	/*multi.LearnA(&rng, nil)
	fmt.Println(multi.A)
	inv := MatrixInverse(multi.A, &rng)
	fmt.Println(inv)
	fmt.Println(multi.A.MulT(inv))
	dist := inv.MulT(points.Sub(multi.U))
	fmt.Println(dist)*/
}
