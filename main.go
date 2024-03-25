// Copyright 2024 The Gauss Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

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
	rng := Rand(1)
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
	multi.LearnA(&rng, nil)
	fmt.Println(multi.E)
	fmt.Println(multi.A)
	inv := MatrixInverse(multi.A, &rng)
	fmt.Println(inv)
	fmt.Println(multi.A.MulT(inv))
}
